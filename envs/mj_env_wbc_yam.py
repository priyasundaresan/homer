import multiprocessing as mp
import time
from threading import Thread

import mujoco
import numpy as np
from ruckig import Result
from scipy.spatial.transform import Rotation as R

from constants import POLICY_CONTROL_PERIOD
from envs.utils.wbc_ik_solver_yam import IKSolver as YamIKSolver
from envs.common_mj_env import (
    ArmController,
    BaseController,
    ShmState,
    ShmImage,
    ShmCameraParameters,
    Renderer,
    MujocoEnvConfig,
    CommonMujocoSim,
    CommonMujocoEnv,
)


class YamMujocoSim(CommonMujocoSim):
    """MuJoCo simulation process for the YAM 6-DOF arm on a mobile base.

    Differences from the Kinova variant (mj_env_wbc.py):
      - 6 arm DOFs (joint1..joint6) instead of 7
      - Gripper controlled by a position actuator (ctrlrange 0–0.041) instead of
        a tendon actuator (ctrlrange 0–255)
      - Arm body name is 'arm' instead of 'gen3/base_link'
      - Gripper joint range is 0.041 instead of 0.8
      - WBC IK uses wbc_ik_solver_yam.IKSolver
    """

    ARM_DOFS = 6
    # left_finger joint range max (used for normalisation)
    GRIPPER_JOINT_RANGE = 0.041
    # ArmController params: inverted so gripper_pos=0 means open (matching Kinova convention)
    # ctrl = offset + scale * gripper_pos  →  0.041 + (-0.041) * gripper_pos
    # gripper_pos=0 (open)  → ctrl=0.041 → finger spreads
    # gripper_pos=1 (closed) → ctrl=0.0   → finger retracts
    GRIPPER_CTRL_OFFSET = 0.041
    GRIPPER_CTRL_SCALE = -0.041

    def __init__(self, task, mjcf_path, command_queue, shm_state, cfg, show_viewer=True):
        super().__init__(
            task, mjcf_path, command_queue, shm_state, show_viewer,
            arm_body_name='arm',
            gripper_joint_range=self.GRIPPER_JOINT_RANGE,
        )
        self.cfg = cfg

        self.wbc_ik_solver = YamIKSolver(
            reset_qpos=self.cfg.arm_reset_qpos,
            base_immobile=self.cfg.base_immobile,
            collision_avoidance=self.cfg.collision_avoidance,
        )

        arm_dofs = self.ARM_DOFS
        self.arm_dofs = arm_dofs

        # qpos/qvel/ctrl slices for the 6-DOF arm
        self.qpos_arm = self.data.qpos[self.base_dofs:(self.base_dofs + arm_dofs)]
        qpos_arm = self.data.qpos[self.base_dofs:(self.base_dofs + arm_dofs)]
        qvel_arm = self.data.qvel[self.base_dofs:(self.base_dofs + arm_dofs)]
        ctrl_arm = self.data.ctrl[self.base_dofs:(self.base_dofs + arm_dofs)]

        # left_finger is the first gripper qpos after the 6 arm joints
        self.qpos_gripper = self.data.qpos[
            (self.base_dofs + arm_dofs):(self.base_dofs + arm_dofs + 1)
        ]
        ctrl_gripper = self.data.ctrl[
            (self.base_dofs + arm_dofs):(self.base_dofs + arm_dofs + 1)
        ]

        self.arm_controller = ArmController(
            qpos_arm, qvel_arm, ctrl_arm,
            self.qpos_gripper, ctrl_gripper,
            self.model.opt.timestep,
            self.cfg.arm_reset_qpos,
            wbc=True,
            arm_dofs=arm_dofs,
            gripper_scale=self.GRIPPER_CTRL_SCALE,
            gripper_offset=self.GRIPPER_CTRL_OFFSET,
        )

        self.reset()
        mujoco.set_mjcb_control(self.control_callback)

    def update_shm_state(self):
        # Call parent to update base_pose, arm_pos, arm_quat, reward, initialized.
        # Then overwrite gripper_pos with the inverted convention so it matches Kinova:
        #   0 = open, 1 = closed
        # (YAM left_finger: 0 = closed, GRIPPER_JOINT_RANGE = open)
        super().update_shm_state()
        self.shm_state.gripper_pos[:] = 1.0 - self.qpos_gripper / self.GRIPPER_JOINT_RANGE

    def reset_task(self):
        if self.task in ["cube", "cube_cam_mounts"]:
            randomized_position = np.random.uniform(
                low=(0.7, -0.2, 0), high=(1.3, 0.2, 0), size=3
            )
            randomized_position[2] = 0.05
            interactive_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "interactive_obj")
            self.data.xpos[interactive_body_id] += randomized_position
            self.data.qpos[
                self.model.joint("interactive_obj_freejoint")
                .id : self.model.joint("interactive_obj_freejoint")
                .id + 3
            ] += randomized_position
        else:
            super().reset_task()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.reset_task()
        mujoco.mj_forward(self.model, self.data)

        self.base_controller.reset()
        self.arm_controller.reset()

        # Explicitly start gripper open (left_finger qpos = max = open for YAM)
        self.qpos_gripper[:] = self.GRIPPER_JOINT_RANGE
        mujoco.mj_forward(self.model, self.data)

        # Update IK solver configuration: base(3) + arm(6) + gripper(2)
        self.wbc_ik_solver.configuration.update(
            self.data.qpos[: self.base_dofs + self.arm_dofs + 2]
        )

    def control_callback(self, *_):
        command = None if self.command_queue.empty() else self.command_queue.get()

        if isinstance(command, tuple) and command[0] == "set_seed":
            self.set_seed(command[1])

        if command == 'reset':
            self.reset()

        if command is not None and 'arm_pos' in command:
            # Solve WBC IK: current qpos = base(3) + arm(6) + zeros(2 gripper)
            full_qpos = self.wbc_ik_solver.solve(
                command['arm_pos'],
                command['arm_quat'],
                np.hstack([self.qpos_base, self.qpos_arm, np.zeros(2)]),
            )
            command['base_pose'] = full_qpos[:3]
            command['arm_qpos'] = full_qpos[3:9]   # 6-DOF arm only

        self.base_controller.control_callback(command)
        self.arm_controller.control_callback(command)
        self.update_shm_state()


class MujocoEnv(CommonMujocoEnv):
    def __init__(self, cfg: MujocoEnvConfig, render_images=True, show_viewer=True, show_images=False):
        super().__init__(cfg, render_images, show_viewer, show_images)

        self.physics_proc = mp.Process(target=self.physics_loop, daemon=True)
        self.physics_proc.start()

    def physics_loop(self):
        sim = YamMujocoSim(
            self.task, self.mjcf_path, self.command_queue, self.shm_state,
            self.cfg, show_viewer=self.show_viewer,
        )
        if self.render_images:
            Thread(target=self.render_loop, args=(sim.model, sim.data), daemon=True).start()
        sim.launch()

    def close(self):
        super().close()
        if self.physics_proc is not None and self.physics_proc.is_alive():
            self.physics_proc.terminate()
            self.physics_proc.join()
            self.physics_proc = None
