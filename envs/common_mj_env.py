import math
import os
import pyrallis
from multiprocessing import get_context as _mp_get_context, Pipe as _mp_Pipe  # for teleop display subprocess
import random
from dataclasses import dataclass, field
import multiprocessing as mp
import time
from multiprocessing import shared_memory
from threading import Thread
from itertools import count
from colorama import Fore, Style
import cv2 as cv
import mujoco
import mujoco.viewer
import numpy as np
from common_utils import Stopwatch
from scipy.spatial.transform import Slerp
from envs.utils.arm_ik_solver import IKSolver
from ruckig import InputParameter, OutputParameter, Result, Ruckig
from scipy.spatial.transform import Rotation as R
from constants import POLICY_CONTROL_PERIOD
from interactive_scripts.dataset_recorder import ActMode, DatasetRecorder, load_episode
from teleop.policies import TeleopPolicy
from envs.utils.camera_utils import make_tf
import common_utils
import pdb


def _teleop_display_worker(conn):
    """Spawned subprocess: displays wrist camera frames via cv2 (works on Mac, not under mjpython)."""
    import cv2
    import numpy as np
    cv2.namedWindow('Wrist View', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Wrist View', 640, 480)
    while True:
        try:
            frame = conn.recv()
        except EOFError:
            break
        if frame is None:
            break
        cv2.imshow('Wrist View', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
    cv2.destroyAllWindows()


class BaseController:
    def __init__(self, qpos, qvel, ctrl, timestep):
        self.qpos = qpos
        self.qvel = qvel
        self.ctrl = ctrl

        # OTG (online trajectory generation)
        num_dofs = 3
        self.last_command_time = None
        self.otg = Ruckig(num_dofs, timestep)
        self.otg_inp = InputParameter(num_dofs)
        self.otg_out = OutputParameter(num_dofs)
        self.otg_inp.max_velocity = [0.5, 0.5, 3.14]
        self.otg_inp.max_acceleration = [0.5, 0.5, 2.36]
        self.otg_res = None

    def reset(self):
        # Initialize base at origin
        self.qpos[:] = np.zeros(3)
        self.ctrl[:] = self.qpos

        # Initialize OTG
        self.last_command_time = time.time()
        self.otg_inp.current_position = self.qpos
        self.otg_inp.current_velocity = self.qvel
        self.otg_inp.target_position = self.qpos
        self.otg_res = Result.Finished

    def control_callback(self, command):
        if command is not None:
            self.last_command_time = time.time()
            if 'base_pose' in command:
                # Set target base qpos
                self.otg_inp.target_position = command['base_pose']
                self.otg_res = Result.Working

        # Maintain current pose if command stream is disrupted
        if time.time() - self.last_command_time > 2.5 * POLICY_CONTROL_PERIOD:
            self.otg_inp.target_position = self.qpos
            self.otg_res = Result.Working

        # Update OTG
        if self.otg_res == Result.Working:
            self.otg_res = self.otg.update(self.otg_inp, self.otg_out)
            self.otg_out.pass_to_input(self.otg_inp)
            self.ctrl[:] = self.otg_out.new_position


class ArmController:
    def __init__(self, qpos, qvel, ctrl, qpos_gripper, ctrl_gripper, timestep, reset_qpos, wbc=False,
                 arm_dofs=7, gripper_scale=255.0, gripper_offset=0.0, ik_solver=None):
        self.qpos = qpos
        self.qvel = qvel
        self.ctrl = ctrl
        self.qpos_gripper = qpos_gripper
        self.ctrl_gripper = ctrl_gripper
        self.reset_qpos = reset_qpos
        self.gripper_scale = gripper_scale
        self.gripper_offset = gripper_offset

        # OTG (online trajectory generation)
        num_dofs = arm_dofs
        self.last_command_time = None
        self.otg = Ruckig(num_dofs, timestep)
        self.otg_inp = InputParameter(num_dofs)
        self.otg_out = OutputParameter(num_dofs)
        if arm_dofs == 7:
            # Kinova Gen3 velocity/acceleration limits
            self.otg_inp.max_velocity = 4 * [math.radians(80)] + 3 * [math.radians(140)]
            self.otg_inp.max_acceleration = 4 * [math.radians(240)] + 3 * [math.radians(450)]
        else:
            # Generic limits for other arm DOF counts (e.g. YAM 6-DOF)
            self.otg_inp.max_velocity = num_dofs * [math.radians(80)]
            self.otg_inp.max_acceleration = num_dofs * [math.radians(240)]
        self.otg_res = None

        self.wbc = wbc
        if not self.wbc:
            self.ik_solver = ik_solver if ik_solver is not None else IKSolver(ee_offset=0.12)

    def reset(self):
        # Initialize arm
        self.qpos[:] = np.array(self.reset_qpos)
        self.ctrl[:] = self.qpos
        self.ctrl_gripper[:] = self.gripper_offset

        # Initialize OTG
        self.last_command_time = time.time()
        self.otg_inp.current_position = self.qpos
        self.otg_inp.current_velocity = self.qvel
        self.otg_inp.target_position = self.qpos
        self.otg_res = Result.Finished

    def control_callback(self, command):
        if command is not None:
            self.last_command_time = time.time()

            if 'arm_qpos' in command:
                # Set target arm qpos
                self.otg_inp.target_position = command['arm_qpos']
                self.otg_res = Result.Working

            elif 'arm_pos' in command:
                # Run inverse kinematics on new target pose
                qpos = self.ik_solver.solve(command['arm_pos'], command['arm_quat'], self.qpos)
                qpos = self.qpos + np.mod((qpos - self.qpos) + np.pi, 2 * np.pi) - np.pi  # Unwrapped joint angles

                # Set target arm qpos
                self.otg_inp.target_position = qpos
                self.otg_res = Result.Working

            if 'gripper_pos' in command:
                # ctrl = offset + scale * gripper_pos
                # Kinova: offset=0,     scale=255   → 0=open,  255=closed
                # YAM:    offset=0.041, scale=-0.041 → 0.041=open, 0=closed
                self.ctrl_gripper[:] = self.gripper_offset + self.gripper_scale * command['gripper_pos']

        # Maintain current pose if command stream is disrupted
        if time.time() - self.last_command_time > 2.5 * POLICY_CONTROL_PERIOD:
            self.otg_inp.target_position = self.otg_out.new_position
            self.otg_res = Result.Working

        # Update OTG
        if self.otg_res == Result.Working:
            self.otg_res = self.otg.update(self.otg_inp, self.otg_out)
            self.otg_out.pass_to_input(self.otg_inp)
            self.ctrl[:] = self.otg_out.new_position


class ShmState:
    def __init__(self, existing_instance=None):
        arr = np.empty(3 + 3 + 4 + 1 + 1 + 1 + 1)
        if existing_instance is None:
            self.shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        else:
            self.shm = shared_memory.SharedMemory(name=existing_instance.shm.name)
        self.data = np.ndarray(arr.shape, buffer=self.shm.buf)
        self.base_pose = self.data[:3]
        self.arm_pos = self.data[3:6]
        self.arm_quat = self.data[6:10]
        self.gripper_pos = self.data[10:11]
        self.initialized = self.data[11:12]
        self.initialized[:] = 0.0
        self.reward = self.data[12:13]
        self.reward[:] = 0
        self.goal_cube = self.data[13:14] # for cube task
        self.goal_cube[:] = 1.0  # default to green cube

    def close(self):
        self.shm.close()

class ShmCameraParameters:
    def __init__(self, existing_instance=None):
        arr = np.empty(4*4 + 3*3)
        if existing_instance is None:
            self.shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        else:
            self.shm = shared_memory.SharedMemory(name=existing_instance.shm.name)
        self.data = np.ndarray(arr.shape, buffer=self.shm.buf)
        self.extrinsics = self.data[:4*4].reshape(4,4)
        self.intrinsics = self.data[4*4:].reshape(3,3)

    def close(self):
        self.shm.close()

class ShmImage:
    def __init__(self, camera_name=None, width=None, height=None, channels=3, existing_instance=None):
        if existing_instance is None:
            self.camera_name = camera_name
            self.channels = channels
            dtype = np.uint8 if channels == 3 else np.float32
            self.shm = shared_memory.SharedMemory(create=True, size=width * height * channels * np.dtype(dtype).itemsize)
            shape = (height, width, channels) if channels == 3 else (height, width)
            self.data = np.ndarray(shape, dtype=dtype, buffer=self.shm.buf)
        else:
            self.camera_name = existing_instance.camera_name
            self.channels = existing_instance.channels
            self.shm = shared_memory.SharedMemory(name=existing_instance.shm.name)
            shape = existing_instance.data.shape
            self.data = np.ndarray(shape, dtype=existing_instance.data.dtype, buffer=self.shm.buf)
        self.data.fill(0)

    def close(self):
        self.shm.close()

    def unlink(self):
        self.shm.unlink()

class Renderer:
    def __init__(self, model, data, shm_image, shm_depth, shm_cam_params):
        self.model = model
        self.data = data
        self.image = np.empty_like(shm_image.data)
        self.depth = np.empty((shm_image.data.shape[0], shm_image.data.shape[1]), dtype=np.float32)  # Depth buffer

        # Attach to existing shared memory image
        self.shm_image = ShmImage(existing_instance=shm_image)
        self.shm_depth = ShmImage(existing_instance=shm_depth)
        self.shm_cam_params = ShmCameraParameters(existing_instance=shm_cam_params)

        # Set up camera
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA.value, shm_image.camera_name)
        self.camera_id = camera_id
        width, height = model.cam_resolution[camera_id]
        self.width = width
        self.height = height
        self.camera = mujoco.MjvCamera()
        self.camera.fixedcamid = camera_id
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.camera_name = shm_image.camera_name

        # Set up context
        self.rect = mujoco.MjrRect(0, 0, width, height)
        self.gl_context = mujoco.gl_context.GLContext(width, height)
        self.gl_context.make_current()
        self.mjr_context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, self.mjr_context)

        # Set up scene
        self.scene_option = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(model, 10000)

    def render(self):
        self.gl_context.make_current()
        mujoco.mjv_updateScene(
            self.model, self.data, self.scene_option, None, self.camera,
            mujoco.mjtCatBit.mjCAT_ALL.value, self.scene
        )
        mujoco.mjr_render(self.rect, self.scene, self.mjr_context)

        # Read RGB and Depth
        mujoco.mjr_readPixels(self.image, self.depth, self.rect, self.mjr_context)

        # Flip the image vertically (MuJoCo's rendering is upside-down)
        self.shm_image.data[:] = np.flipud(self.image)

        extent = self.model.stat.extent
        near = self.model.vis.map.znear * extent
        far = self.model.vis.map.zfar * extent
        image = near / (1 - self.depth * (1 - near / far))

        #z_near = self.model.vis.map.znear
        #z_far = self.model.vis.map.zfar
        #self.depth = (z_near * z_far) / (z_far - (z_far - z_near) * self.depth)
        self.depth = near / (1 - self.depth * (1 - near / far))

        # Store depth image in shared memory (also flip it)
        self.shm_depth.data[:] = np.flipud(self.depth)

    def get_params(self):
        aspect_ratio = self.width / self.height
        fovy = np.radians(self.model.cam_fovy[self.camera_id])
        fovx = 2 * np.arctan(np.tan(fovy / 2) * aspect_ratio)
        fx, fy = self.width / (2 * np.tan(fovx / 2)), self.height / (2 * np.tan(fovy / 2))
        cx, cy = self.width / 2, self.height / 2

        intrinsics = np.array([
            [fx, 0, cx],  # Intrinsics for x-axis
            [0, fy, cy],  # Intrinsics for y-axis
            [0, 0, 1]  # Homogeneous coordinates
        ])

        cam_pos = self.model.cam_pos[self.camera_id]  # Local camera position
        cam_quat = self.model.cam_quat[self.camera_id]  # Local quaternion

        # Convert quaternion to rotation matrix
        cam_rot = np.zeros((9,))
        mujoco.mju_quat2Mat(cam_rot, cam_quat)
        cam_rot = cam_rot.reshape((3, 3))

        # Find parent body of the camera
        body_id = self.model.cam_bodyid[self.camera_id]  # Body that the camera is attached to
        body_pos = self.data.xpos[body_id]  # Global position of the body
        body_quat = self.data.xquat[body_id]  # Global quaternion of the body

        # Convert body quaternion to rotation matrix
        body_rot = np.zeros((9,))
        mujoco.mju_quat2Mat(body_rot, body_quat)
        body_rot = body_rot.reshape((3, 3))

        # Compute global camera transform
        global_cam_pos = body_pos + body_rot @ cam_pos
        global_cam_rot = body_rot @ cam_rot

        # Construct global extrinsic matrix
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = global_cam_rot
        extrinsics[:3, 3] = global_cam_pos
        self.shm_cam_params.data[:] = np.hstack((extrinsics.flatten(), intrinsics.flatten()))

    def close(self):
        self.gl_context.free()
        self.gl_context = None
        self.mjr_context.free()
        self.mjr_context = None

class CommonMujocoSim:
    def __init__(self, task, mjcf_path, command_queue, shm_state, show_viewer=True,
                 arm_body_name='gen3/base_link', gripper_joint_range=0.8):
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.model.vis.map.znear = 0.05
        self.model.vis.map.zfar = 8.0
        self.data = mujoco.MjData(self.model)
        self.command_queue = command_queue
        self.show_viewer = show_viewer

        self.task = task
        assert self.task in ["cube", "cube_cam_mounts", "cube_size", "cube_distractor", "cube_specified", "open", "dishwasher", "plate_dishrack"]

        # Enable gravity compensation for robot bodies only; task objects fall normally.
        # We walk the full body tree and disable gravcomp for any body that is a
        # descendant of 'interactive_obj' (the free-floating task object root).
        self.model.body_gravcomp[:] = 1.0
        body_names = {self.model.body(i).name for i in range(self.model.nbody)}
        object_roots = ['interactive_obj']
        for root_name in object_roots:
            if root_name not in body_names:
                continue
            # Disable gravcomp for the root and all its descendants
            root_id = self.model.body(root_name).id
            for i in range(self.model.nbody):
                body = self.model.body(i)
                # Walk up the parent chain to see if this body is under root_id
                bid = i
                while bid != 0:
                    if bid == root_id:
                        self.model.body_gravcomp[i] = 0.0
                        break
                    bid = self.model.body(bid).parentid

        # Cache references to array slices
        self.base_dofs = base_dofs = self.model.body('base_link').jntnum.item()
        self.qpos_base = self.data.qpos[:base_dofs]
        qvel_base = self.data.qvel[:base_dofs]
        ctrl_base = self.data.ctrl[:base_dofs]

        # Controllers
        self.base_controller = BaseController(self.qpos_base, qvel_base, ctrl_base, self.model.opt.timestep)

        # Shared memory state for observations
        self.shm_state = ShmState(existing_instance=shm_state)

        # Variables for calculating arm pos and quat in the base-local frame
        site_id = self.model.site('pinch_site').id
        self.site_xpos = self.data.site(site_id).xpos
        self.site_xmat = self.data.site(site_id).xmat
        self.site_quat = np.empty(4)
        self.base_height = self.model.body(arm_body_name).pos[2]
        self.arm_forward = self.model.body(arm_body_name).pos[0]
        self.base_rot_axis = np.array([0.0, 0.0, 1.0])
        self.base_quat_inv = np.empty(4)
        # Range of the primary gripper joint (used to normalise gripper_pos to [0, 1])
        self.gripper_joint_range = gripper_joint_range

    def update_shm_state(self):
        # Update base pose
        self.shm_state.base_pose[:] = self.qpos_base

        # Update arm pos
        # self.shm_state.arm_pos[:] = self.site_xpos
        site_xpos = self.site_xpos.copy()
        site_xpos[2] -= self.base_height  # Base height offset
        site_xpos[:2] -= self.qpos_base[:2]  # Base position inverse
        mujoco.mju_axisAngle2Quat(self.base_quat_inv, self.base_rot_axis, -self.qpos_base[2])  # Base orientation inverse
        mujoco.mju_rotVecQuat(self.shm_state.arm_pos, site_xpos, self.base_quat_inv)  # Arm pos in local frame
        self.shm_state.arm_pos[0] -= self.arm_forward

        # Update arm quat
        mujoco.mju_mat2Quat(self.site_quat, self.site_xmat)
        # self.shm_state.arm_quat[:] = self.site_quat
        mujoco.mju_mulQuat(self.shm_state.arm_quat, self.base_quat_inv, self.site_quat)  # Arm quat in local frame

        # Update gripper pos (normalise to [0, 1] using the robot-specific joint range)
        self.shm_state.gripper_pos[:] = self.qpos_gripper / self.gripper_joint_range

        # Notify reset() function that state has been initialized
        self.shm_state.initialized[:] = 1.0

        # Check success and update reward
        self.shm_state.reward[:] = self.is_success()

    def reset_task(self):

        ## Task specific randomizations
        if self.task in ["cube", "cube_cam_mounts"]:
            randomized_position = np.random.uniform(
                low=(0.5, -0.2, 0), high=(1.3, 0.2, 0), size=3
            )
            randomized_position[2] = 0.05  # drop it from 5cm above ground plane
            interactive_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "interactive_obj")
            self.data.xpos[interactive_body_id] += randomized_position
            self.data.qpos[
                self.model.joint("interactive_obj_freejoint")
                .id : self.model.joint("interactive_obj_freejoint")
                .id
                + 3
            ] += randomized_position

        elif self.task == "cube_size":
            randomized_position = np.random.uniform(
                low=(0.5, -0.2, 0), high=(1.0, 0.2, 0), size=3
            )
            randomized_position[2] = 0.05  # drop it from 5cm above ground plane
            interactive_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "interactive_obj")
            self.data.xpos[interactive_body_id] += randomized_position
            self.data.qpos[
                self.model.joint("interactive_obj_freejoint")
                .id : self.model.joint("interactive_obj_freejoint")
                .id
                + 3
            ] += randomized_position
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
            size = np.random.uniform(low=[0.02, 0.01, 0.01], high=[0.04, 0.0275, 0.0275], size=3)
            self.model.geom_size[geom_id] = size

        elif self.task == "cube_distractor":
            # Define bounds
            cube_low = np.array([0.5, -0.2, 0.1])
            cube_high = np.array([1.0, 0.2, 0.1])  # Cube restricted to x in [0.5, 0.65]
            
            distractor_low = np.array([0.6, -0.2, 0.1])
            distractor_high = np.array([0.9, 0.2, 0.1])  # Distractors x in [0.7, 0.9]
            
            min_distance = 0.05
            
            # Sample cube position
            cube_pos = np.random.uniform(low=cube_low, high=cube_high)
            
            # Sample 3 distractor positions
            distractor_positions = []
            for _ in range(3):
                for _ in range(500):
                    candidate = np.random.uniform(low=distractor_low, high=distractor_high)
                    if all(np.linalg.norm(candidate[:2] - pos[:2]) >= min_distance for pos in distractor_positions):
                        distractor_positions.append(candidate)
                        break
                else:
                    raise RuntimeError("Failed to sample distractors with required separation.")
            
            # Combine into list: cube first, then distractors
            positions = [cube_pos] + distractor_positions
            
            # Body names and joint names
            bodies_and_joints = [
                ("interactive_obj", "interactive_obj_freejoint"),
                ("distractor1", "distractor1_freejoint"),
                ("distractor2", "distractor2_freejoint"),
                ("distractor3", "distractor3_freejoint"),
            ]
            
            # Move them
            for pos, (body_name, joint_name) in zip(positions, bodies_and_joints):
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                qpos_adr = self.model.jnt_qposadr[joint_id]
                self.data.qpos[qpos_adr : qpos_adr + 3] = pos
            
            # Finalize
            mujoco.mj_forward(self.model, self.data)

        elif self.task == "cube_specified":
            # Define bounds
            low = np.array([0.5, -0.2, 0.1])  # x, y bounds, z = 0.1 fixed
            high = np.array([0.9, 0.2, 0.1])
            min_distance = 0.02  # 2 cm
            
            # Sample first cube
            cube1 = np.random.uniform(low=low, high=high)
            # Sample second cube
            for _ in range(100):
                cube2 = np.random.uniform(low=low, high=high)
                if np.linalg.norm(cube2[:2] - cube1[:2]) >= min_distance:
                    break
            else:
                raise RuntimeError("Failed to sample two cubes with required separation.")
            
            # Get joint IDs
            joint1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "interactive_obj_freejoint")
            joint2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "interactive_obj2_freejoint")
            
            # Get qpos addresses
            qpos1_adr = self.model.jnt_qposadr[joint1_id]
            qpos2_adr = self.model.jnt_qposadr[joint2_id]
            
            # Set cube1 position
            self.data.qpos[qpos1_adr : qpos1_adr + 3] = cube1
            # Set cube2 position
            self.data.qpos[qpos2_adr : qpos2_adr + 3] = cube2
            
            # Then tell MuJoCo to update everything!
            mujoco.mj_forward(self.model, self.data)

        elif self.task == "open":
            #randomized_position = np.random.uniform(
            #    low=(0.1, -0.5, 0), high=(0.7, 0.5, 0), size=3
            #)
            randomized_position = np.random.uniform(
                low=(0.1, -0.5, 0), high=(0.8, 0.5, 0), size=3
            )
            interactive_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "interactive_obj")
            self.data.xpos[interactive_body_id] += randomized_position
            self.data.qpos[
                self.model.joint("interactive_obj_freejoint")
                .id : self.model.joint("interactive_obj_freejoint")
                .id
                + 3
            ] += randomized_position

        elif self.task == "dishwasher":
            randomized_position = np.random.uniform(
                low=(0.1, -0.5, 0), high=(0.7, 0.5, 0), size=3
            )
            interactive_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "interactive_obj")
            self.data.xpos[interactive_body_id] += randomized_position
            self.data.qpos[
                self.model.joint("interactive_obj_freejoint")
                .id : self.model.joint("interactive_obj_freejoint")
                .id
                + 3
            ] += randomized_position

        elif self.task == "plate_dishrack":
            # Randomize plate position on the table, in front of the dishrack.
            # Table surface at z=0.80; plate half-height=0.003 so center at z=0.803.
            # Dishrack fixed at x=0.90, y=0; its open face is at x≈0.71.
            plate_pos = np.array([
                np.random.uniform(0.5, 0.80),
                np.random.uniform(-0.15, -0.35),  # -y side; rack is on +y side
                0.90,   # drop from 10cm above table surface (z=0.80) so it settles cleanly
            ])
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "interactive_obj_freejoint")
            qpos_adr = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qpos_adr : qpos_adr + 3] = plate_pos
            mujoco.mj_forward(self.model, self.data)

    def is_success(self):
        if self.task in ["cube", "cube_cam_mounts", "cube_size", "cube_distractor"]:
            ### Check whether the cube is lifted off the floor by 10cm
            interactive_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "interactive_obj")
            cube_pos = self.data.xpos[interactive_body_id]
            z_thresh = 0.10
            reward = (cube_pos[2] > z_thresh)
        elif self.task in ["cube_specified"]:
            ### Check whether the cube is lifted off the floor by 10cm
            goal_is_green = (self.shm_state.goal_cube[0] == 1.0)
            id_name = "interactive_obj" if goal_is_green else "interactive_obj2"
            interactive_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, id_name)
            cube_pos = self.data.xpos[interactive_body_id]
            z_thresh = 0.10
            reward = (cube_pos[2] > z_thresh)
        elif self.task == "open":
            door_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "rightdoorhinge")
            right_door_angle = self.data.sensordata[door_id]
            angle_thresh = 0.5
            reward = right_door_angle > angle_thresh
        elif self.task == "dishwasher":
            door_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "door")
            door_angle = self.data.sensordata[door_id]
            assert -np.pi / 2 < door_angle < 0.1
            angle_thresh = -np.pi / 8
            reward = door_angle < angle_thresh
        elif self.task == "plate_dishrack":
            # Success when plate center is inside the dishrack interior.
            # Dishrack at x=0.90, y=0, z=0.80. Rack floor at z=0.82; plate center in rack ~0.823.
            plate_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "interactive_obj")
            plate_pos = self.data.xpos[plate_body_id]
            in_rack = (
                0.72 < plate_pos[0] < 1.08
                and 0.03 < plate_pos[1] < 0.37   # rack at y=0.20, interior y in (0.03, 0.37)
                and plate_pos[2] > 0.850
            )
            reward = float(in_rack)
        return reward

    def reset(self):
        pass

    def set_seed(self, seed):
        print('Seeding with: %d'%seed)
        np.random.seed(seed)
        random.seed(seed)

    def launch(self):
        if self.show_viewer:
            mujoco.viewer.launch(self.model, self.data, show_left_ui=False, show_right_ui=False)

        else:
            # Run headless simulation at real-time speed
            last_step_time = 0
            while True:
                while time.time() - last_step_time < self.model.opt.timestep:
                    time.sleep(0.0001)
                last_step_time = time.time()
                try:
                    mujoco.mj_step(self.model, self.data)
                except:
                    import pdb; pdb.set_trace()

@dataclass
class MujocoEnvConfig:
    wbc: int
    cameras: list[str]
    pcl_cameras: list[str]
    task: str
    data_folder: str
    is_sim: int = 1
    robot: str = "kinova"  # "kinova" or "yam"
    arm_reset_qpos: list[float] = field(default_factory=lambda: [
        0.0,
        -0.34906585,
        3.14159265,
        -2.54818071,
        0.0,
        -0.87266463,
        1.57079633,
    ])
    min_bound: list[float] = field(default_factory=list)
    max_bound: list[float] = field(default_factory=list)
    # WBC IK solver options
    base_immobile: bool = False
    collision_avoidance: bool = False

class CommonMujocoEnv:
    def __init__(self, cfg: MujocoEnvConfig, render_images=True, show_viewer=True, show_images=False):
        self.cfg = cfg

        self.ARM_BASE_OFFSET = [0.1199, 0, 0.3948]

        self.render_images = render_images
        self.show_viewer = show_viewer
        self.show_images = show_images
        self.command_queue = mp.Queue(1)
        self.num_step = 0
        self.reward = 0

        self.task = self.cfg.task
        
        assert self.task in ["cube", "cube_cam_mounts", "cube_size", "cube_distractor", "cube_specified", "open", "dishwasher", "plate_dishrack"]
        if self.task in ["cube", "cube_cam_mounts", "cube_size", "cube_distractor", "cube_specified"]:
            self.max_num_step = 325
        elif self.task == "open":
            self.max_num_step = 800
        elif self.task == "dishwasher":
            self.max_num_step = 1000
        elif self.task == "plate_dishrack":
            self.max_num_step = 500

        robot = getattr(self.cfg, 'robot', 'kinova')
        TASK_TO_MJCF_PATH = {
            # Kinova Gen3 + 2f85 gripper
            ('cube',          'kinova'): "mj_assets/stanford_tidybot2/cube.xml",
            ('cube_cam_mounts','kinova'): "mj_assets/stanford_tidybot2/cube_cam_mounts.xml",
            ('cube_size',     'kinova'): "mj_assets/stanford_tidybot2/cube_size.xml",
            ('cube_distractor','kinova'): "mj_assets/stanford_tidybot2/cube_distractor.xml",
            ('cube_specified', 'kinova'): "mj_assets/stanford_tidybot2/cube_specified.xml",
            ('open',          'kinova'): "mj_assets/stanford_tidybot2/open.xml",
            ('dishwasher',    'kinova'): "mj_assets/stanford_tidybot2/dishwasher.xml",
            ('plate_dishrack','kinova'): "mj_assets/stanford_tidybot2/plate_dishrack.xml",
            # YAM arm
            ('cube',          'yam'):    "mj_assets/stanford_tidybot2/cube_yam.xml",
            ('dishwasher',    'yam'):    "mj_assets/stanford_tidybot2/dishwasher_yam.xml",
            ('open',          'yam'):    "mj_assets/stanford_tidybot2/open_yam.xml",
        }
        self.mjcf_path = TASK_TO_MJCF_PATH[(self.cfg.task, robot)]

        self.shm_state = ShmState()
        self.shm_cam_params = []
        self.shm_images_rgb = []
        self.shm_images_depth = []
        self.camera_names = []

        self.data_folder = cfg.data_folder
        self.recorder = DatasetRecorder(self.data_folder)
        self.stopwatch = Stopwatch()
        self.teleop_policy = None

        if self.render_images:
            for camera_name in self.cfg.cameras:
                self.shm_cam_params.append(ShmCameraParameters())
                self.camera_names.append(camera_name)
                self.shm_images_rgb.append(ShmImage(camera_name, 640, 480, channels=3))
                self.shm_images_depth.append(ShmImage(camera_name, 640, 480, channels=1))

        if self.render_images and self.show_images:
            # Start visualizer loop
            self.visualizer_process = mp.Process(target=self.visualizer_loop, daemon=True)
            self.visualizer_process.start()

        # Will be lazily started on first collect_episode (after physics_proc is spawned)
        self._display_conn = None
        self._display_proc = None

    def _dump_or_check_env_cfg(self):
        cfg_path = os.path.join(self.data_folder, "env_cfg.yaml")
        if not os.path.exists(cfg_path):
            print(f"saving env cfg to {cfg_path}")
            pyrallis.dump(self.cfg, open(cfg_path, "w"))  # type: ignore
        else:
            assert common_utils.check_cfg(MujocoEnvConfig, cfg_path, self.cfg), \
                f"Error: {self.data_folder} contains a different config than the current one"

    def physics_loop(self):
        pass

    def render_loop(self, model, data):
        # Set up renderers
        renderers = [Renderer(model, data, shm_image, shm_depth_image, shm_cam_params) for (shm_image, shm_depth_image, shm_cam_params) in zip(self.shm_images_rgb, self.shm_images_depth, self.shm_cam_params)]

        # Render camera images continuously
        while True:
            start_time = time.time()
            for renderer in renderers:
                renderer.render()
                renderer.get_params()
            render_time = time.time() - start_time
            if render_time > 0.1:  # 10 fps
                print(f'Warning: Offscreen rendering took {1000 * render_time:.1f} ms, try making the Mujoco viewer window smaller to speed up offscreen rendering')

    def visualizer_loop(self):
        shm_images = [ShmImage(existing_instance=shm_image) for shm_image in self.shm_images_rgb]
        last_imshow_time = time.time()
        while True:
            while time.time() - last_imshow_time < 0.1:  # 10 fps
                time.sleep(0.01)
            last_imshow_time = time.time()
            for i, shm_image in enumerate(shm_images[1:]):
                image = shm_image.data
                resized_image = cv.resize(image, (480 * 2, 360 * 2))
                cv.imshow(shm_image.camera_name, cv.cvtColor(resized_image, cv.COLOR_RGB2BGR))
                if i < 3:
                    cv.moveWindow(shm_image.camera_name, 480 * 2 * i, -100)
                else:
                    cv.moveWindow(shm_image.camera_name, 480 * 2 * (i - 3), 480 * 2)
            cv.waitKey(1)

    def reset(self):
        self.num_step = 0
        self.recorder.get_next_idx()
        self.shm_state.initialized[:] = 0.0
        self.command_queue.put('reset')
        while self.shm_state.initialized == 0.0:
            time.sleep(0.01)

        if self.render_images:
            while any(np.all(shm_image.data == 0) for shm_image in self.shm_images_rgb):
                time.sleep(0.01)

        if self.task == 'cube_specified':
            self.goal_cube = random.choice(["green", "red"])
            self.shm_state.goal_cube[:] = 1.0 if self.goal_cube == "green" else 0.0
            return self.goal_cube
        elif 'cube' in self.task:
            self.goal_cube = "green"
            return self.goal_cube
        return None

    def get_obs(self):
        base_pose = self.shm_state.base_pose.copy()
        arm_pos = self.shm_state.arm_pos.copy()
        arm_quat = self.shm_state.arm_quat[[1, 2, 3, 0]]  # (w, x, y, z) -> (x, y, z, w)
        if arm_quat[3] < 0.0:  # Enforce quaternion uniqueness
            np.negative(arm_quat, out=arm_quat)

        arm_pos_global = self.local_to_global_arm_pos(arm_pos, base_pose)

        gripper_pos = self.shm_state.gripper_pos.copy()
        obs = {
            'base_pose': base_pose,
            'arm_pos': arm_pos_global,
            'arm_quat': arm_quat,
            'gripper_pos': gripper_pos,
            'reward': self.shm_state.reward.copy(),
            'proprio': np.hstack((arm_pos, arm_quat, gripper_pos, base_pose))
        }

        if self.render_images:
            camera_axis_correction = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            for (name, shm_rgb, shm_depth, shm_cam_params) in zip(self.camera_names, self.shm_images_rgb, self.shm_images_depth, self.shm_cam_params):
                obs[f'{name}_image'] = shm_rgb.data.copy()
                obs[f'{name}_depth'] = shm_depth.data.copy()
                cam_params = shm_cam_params.data.copy()
                obs["%s_T" % name] = cam_params[:4*4].reshape((4,4)) @ camera_axis_correction
                obs["%s_K" % name] = cam_params[4*4:].reshape((3,3))

        return obs

    def seed(self, seed):
        self.command_queue.put(("set_seed", seed))
        time.sleep(0.1)

    def step(self, action):
        self.command_queue.put(action)
        self.num_step += 1

    def close(self):
        self.shm_state.close()
        self.shm_state.shm.unlink()

        if self.render_images:
            for shm_rgb, shm_depth, shm_cam_param in zip(self.shm_images_rgb, self.shm_images_depth, self.shm_cam_params):
                shm_rgb.close()
                shm_rgb.shm.unlink()
                shm_depth.close()
                shm_depth.shm.unlink()
                shm_cam_param.close()
                shm_cam_param.shm.unlink()

        if hasattr(self, 'visualizer_process') and self.visualizer_process.is_alive():
            self.visualizer_process.terminate()
            self.visualizer_process.join()

    def local_to_global_arm_pos(self, arm_pos_local, base_pose):
        T_base_world = np.eye(4)
        T_base_world[:3, :3] = R.from_euler('z', base_pose[2]).as_matrix()
        T_base_world[:3, 3] = np.array([base_pose[0], base_pose[1], 0]) + self.ARM_BASE_OFFSET
        arm_pos_global = T_base_world@np.array([arm_pos_local[0], arm_pos_local[1], arm_pos_local[2], 1.0])
        arm_pos_global = arm_pos_global[:3]
        return arm_pos_global

    def global_to_local_arm_pos(self, arm_pos_global, base_pose):
        T_base_world = np.eye(4)
        T_base_world[:3, :3] = R.from_euler('z', base_pose[2]).as_matrix()
        T_base_world[:3, 3] = np.array([base_pose[0], base_pose[1], 0]) + self.ARM_BASE_OFFSET
        arm_pos_local = np.linalg.inv(T_base_world)@np.array([arm_pos_global[0], arm_pos_global[1], arm_pos_global[2], 1.0])
        arm_pos_local = arm_pos_local[:3]
        return arm_pos_local

    def collect_episode(self):
        self._dump_or_check_env_cfg()
        # Reset
        if self.teleop_policy is None:
            self.teleop_policy = TeleopPolicy()

        # Seed based on episode idx
        print(Fore.BLUE + "Recording episode: %d" % self.recorder.episode_idx + Style.RESET_ALL)
        self.seed(self.recorder.episode_idx)
        self.reset()
        self.teleop_policy.reset()

        episode_ended = False
        start_time = time.time()

        # Spawn wrist display process once (kept alive across episodes)
        if self._display_conn is None and self.render_images and 'wrist' in self.camera_names:
            _parent_conn, _child_conn = _mp_Pipe()
            _ctx = _mp_get_context('spawn')
            self._display_proc = _ctx.Process(target=_teleop_display_worker, args=(_child_conn,), daemon=True)
            self._display_proc.start()
            _child_conn.close()
            self._display_conn = _parent_conn

        prev_obs = self.get_obs()  # Only capture observations at 10Hz

        for step_idx in count():
            # Enforce desired control freq
            step_end_time = start_time + step_idx * POLICY_CONTROL_PERIOD
            while time.time() < step_end_time:
                time.sleep(0.0001)

            # Get latest observation
            obs = self.get_obs()

            # Send wrist frame to persistent display subprocess
            if self._display_conn is not None and 'wrist_image' in obs:
                try:
                    self._display_conn.send(obs['wrist_image'])
                except BrokenPipeError:
                    self._display_conn = None

            # Get action
            processed_obs = obs.copy()
            if not self.cfg.wbc:
                processed_obs['arm_pos'] = self.global_to_local_arm_pos(obs['arm_pos'], obs['base_pose'])
            action = self.teleop_policy.step(processed_obs)

            # No action if teleop not enabled
            if action is None:
                prev_obs = obs
                continue

            # Execute valid action on robot
            if isinstance(action, dict):
                self.step(action)

                action_quat = action["arm_quat"]
                if action_quat[3] < 0.0:
                    np.negative(action_quat, out=action_quat)

                record_action_arm_pos = action["arm_pos"]
                if not self.cfg.wbc:
                    record_action_arm_pos = self.local_to_global_arm_pos(action["arm_pos"], obs["base_pose"])

                record_action = np.concatenate(
                    [
                        record_action_arm_pos,
                        action_quat,
                        action["gripper_pos"],
                        action["base_pose"]
                    ]
                )

                delta_pos = record_action[:3] - prev_obs['arm_pos']
                delta_rot = R.from_quat(action["arm_quat"]) * R.from_quat(prev_obs["arm_quat"]).inv()
                delta_quat = delta_rot.as_quat()
                delta_base_pose = record_action[-3:] - prev_obs["base_pose"]

                # Delta action to record
                record_delta_action = np.concatenate(
                    [delta_pos, delta_quat, action["gripper_pos"], delta_base_pose]
                )

                if obs['reward']:
                    print(Fore.GREEN + "Success" + Style.RESET_ALL)
                else:
                    print(Fore.LIGHTBLACK_EX + str(record_delta_action.round(2)) + Style.RESET_ALL)

                if not episode_ended:
                    # Record executed action
                    self.recorder.record(
                        ActMode.Dense,
                        obs,
                        action=record_action,
                        delta_action=record_delta_action,
                        teleop_mode=action["teleop_mode"]
                    )

            # Episode ended
            elif not episode_ended and action == 'end_episode':
                episode_ended = True
                self.recorder.end_episode(save=True)
                print('Teleop is now active. Press "Reset env" in the web app when ready.')

            # Ready for env reset
            elif action == 'reset_env':
                break

            prev_obs = obs

    def replay_episode(self, episode_fn, replay_mode="absolute"):
        self._dump_or_check_env_cfg()
        assert(replay_mode in ["absolute", "delta"])
        demo = load_episode(episode_fn)

        # Reset and seed based on episode idx
        self.seed(int(os.path.splitext(os.path.basename(episode_fn))[0][len("demo"):]))
        self.reset()

        start_time = time.time()

        for step_idx, step in enumerate(list(demo)):
            # Enforce desired control freq
            step_end_time = start_time + step_idx * POLICY_CONTROL_PERIOD
            while time.time() < step_end_time:
                time.sleep(0.0001)

            recorded_obs = step["obs"]

            ### Absolute Replay
            if replay_mode == 'absolute':
                recorded_action = step["action"]

                if not self.cfg.wbc:
                    arm_pos_action = self.global_to_local_arm_pos(recorded_action[:3], recorded_obs["base_pose"])
                    base_action = recorded_action[8:11]
                else:
                    arm_pos_action = recorded_action[:3]
                    base_action = np.zeros(3)

                action = {
                    "base_pose": base_action,
                    "arm_pos": arm_pos_action,
                    "arm_quat": recorded_action[3:7],
                    "gripper_pos": np.array(recorded_action[7]),
                }

            ### Delta Replay
            else:
                recorded_delta_action = step['delta_action']

                if not self.cfg.wbc:
                    arm_pos_action = self.global_to_local_arm_pos(recorded_obs["arm_pos"], recorded_obs["base_pose"]) + recorded_delta_action[:3]
                    base_action = recorded_obs["base_pose"] + recorded_delta_action[8:11]
                else:
                    arm_pos_action = recorded_obs["arm_pos"] + recorded_delta_action[:3]
                    base_action = np.zeros(3)

                action_rot = R.from_quat(recorded_delta_action[3:7]) * R.from_quat(recorded_obs["arm_quat"])
                action_quat = action_rot.as_quat()
                action_quat = action_quat/np.linalg.norm(action_quat)

                if action_quat[3] < 0.0:  # Enforce quaternion uniqueness
                    np.negative(action_quat, out=action_quat)

                action = {
                   'base_pose': base_action,
                   'arm_pos': arm_pos_action,
                   'arm_quat': action_quat,
                   'gripper_pos': np.array(recorded_delta_action[7]),
                }

            self.step(action)

    def move_to_base_waypoint(self, target_base_pose, threshold_pos=0.01, threshold_theta=0.01, recorder=None):
        """
        Moves the robot base smoothly to a target [x, y, theta] pose.

        Args:
            target_base_pose (array-like): [x, y, theta] target for the base.
            threshold_pos (float): Position error threshold for stopping.
            threshold_theta (float): Rotation error threshold (in radians) for stopping.

        Returns:
            bool: True if the target is reached.
        """
        obs = self.get_obs()
        curr_base_pose = obs["base_pose"]

        while True:
            obs = self.get_obs()
            curr_base_pose = obs["base_pose"]

            # Stop when both position and rotation errors are small
            pos_error_norm = np.linalg.norm(curr_base_pose[:2] - target_base_pose[:2])
            theta_error = target_base_pose[2] - curr_base_pose[2]

            if pos_error_norm < threshold_pos and abs(theta_error) < threshold_theta:
                return True, pos_error_norm  # Target reached

            # Execute action
            self.step({"base_pose": target_base_pose})
            time.sleep(POLICY_CONTROL_PERIOD)  # Maintain control rate

            if recorder is not None:
                recorder.add_numpy(obs, ["viewer_image"])

        return False, pos_error_norm

    def move_to_arm_waypoint(self, target_arm_pos, target_arm_quat, target_gripper_pos, step_size=0.1, threshold_pos=0.01, threshold_quat=0.01, recorder=None):
        """
        Moves the robot arm towards a target position and orientation using interpolation.

        Args:
            target_arm_pos (array-like): [x, y, z] target for the arm end-effector.
            target_arm_quat (array-like): [x, y, z, w] target quaternion for arm orientation.
            target_gripper_pos (float): Target gripper position (0.0 closed, 1.0 open).
            step_size (float): Maximum step size per iteration.
            threshold_pos (float): Position error threshold for stopping.
            threshold_quat (float): Quaternion error threshold for stopping.

        Returns:
            bool: True if the target is reached.
        """

        # Ensure consistent quaternion sign
        if target_arm_quat[3] < 0:
            np.negative(target_arm_quat, out=target_arm_quat)

        reached = False
        pos_error_norm = np.inf
        MAX_STEP = 50
        step = 0

        while not reached:
            # Get current position and orientation
            obs = self.get_obs()

            curr_arm_pos, curr_arm_quat = obs["arm_pos"], obs["arm_quat"]

            # Compute position error
            pos_error = target_arm_pos - curr_arm_pos
            pos_error_norm = np.linalg.norm(pos_error)
            print(step, pos_error_norm)

            # Compute quaternion error
            quat_error = 1 - abs(np.dot(curr_arm_quat, target_arm_quat))

            if pos_error_norm < threshold_pos and quat_error < threshold_quat:
                reached = True
                break

            elif step > MAX_STEP:
                break

            # Compute interpolated position step
            step_vec = step_size * pos_error / (pos_error_norm + 1e-6)  # Avoid division by zero
            next_pos = curr_arm_pos + step_vec if pos_error_norm > step_size else target_arm_pos

            if not self.cfg.wbc:
                next_pos = self.global_to_local_arm_pos(next_pos, obs['base_pose'])

            # Compute interpolated quaternion step using Slerp
            key_times = [0, 1]  # Define key times
            key_rots = R.from_quat([curr_arm_quat, target_arm_quat])  # Define key rotations
            slerp = Slerp(key_times, key_rots)  # Create Slerp object
            interp_ratio = min(step_size / (pos_error_norm + 1e-6), 1.0)  # Normalize step size
            next_quat = slerp([interp_ratio]).as_quat()[0]  # Interpolated quaternion

            # Execute action
            self.step({
                "arm_pos": next_pos,
                "arm_quat": next_quat,
                "gripper_pos": 1 if obs['gripper_pos'] > 0.3 else obs['gripper_pos'],
            })

            time.sleep(POLICY_CONTROL_PERIOD)  # Maintain control rate
            step += 1

            if recorder is not None:
                recorder.add_numpy(obs, ["viewer_image"])

        # Move the gripper
        for _ in range(10):  # Hack: Execute gripper action for 10 timesteps
            obs = self.get_obs()
            self.step({"gripper_pos": target_gripper_pos})
            time.sleep(POLICY_CONTROL_PERIOD)

            if recorder is not None:
                recorder.add_numpy(obs, ["viewer_image"])

        return reached, pos_error_norm
