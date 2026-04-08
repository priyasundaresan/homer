import multiprocessing as mp
import time
from threading import Thread

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from constants import POLICY_CONTROL_PERIOD
from envs.utils.arm_ik_solver_yam import IKSolver as YamArmIKSolver
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


class YamBaseArmSim(CommonMujocoSim):
    """MuJoCo simulation process for the YAM 6-DOF arm on a mobile base, base-arm mode.

    Base and arm move independently (no whole-body IK). Arm IK is solved in the
    base-local frame using a Jacobian damped-least-squares solver.
    """

    ARM_DOFS = 6
    GRIPPER_JOINT_RANGE = 0.041
    GRIPPER_CTRL_OFFSET = 0.041
    GRIPPER_CTRL_SCALE = -0.041

    def __init__(self, task, mjcf_path, command_queue, shm_state, cfg, show_viewer=True):
        super().__init__(
            task, mjcf_path, command_queue, shm_state, show_viewer,
            arm_body_name='arm',
            gripper_joint_range=self.GRIPPER_JOINT_RANGE,
        )
        self.cfg = cfg

        arm_dofs = self.ARM_DOFS
        self.arm_dofs = arm_dofs

        self.qpos_arm = self.data.qpos[self.base_dofs:(self.base_dofs + arm_dofs)]
        qpos_arm = self.data.qpos[self.base_dofs:(self.base_dofs + arm_dofs)]
        qvel_arm = self.data.qvel[self.base_dofs:(self.base_dofs + arm_dofs)]
        ctrl_arm = self.data.ctrl[self.base_dofs:(self.base_dofs + arm_dofs)]

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
            wbc=False,
            arm_dofs=arm_dofs,
            gripper_scale=self.GRIPPER_CTRL_SCALE,
            gripper_offset=self.GRIPPER_CTRL_OFFSET,
            ik_solver=YamArmIKSolver(),
        )

        self.reset()
        mujoco.set_mjcb_control(self.control_callback)

    def update_shm_state(self):
        super().update_shm_state()
        self.shm_state.gripper_pos[:] = 1.0 - self.qpos_gripper / self.GRIPPER_JOINT_RANGE

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.reset_task()
        mujoco.mj_forward(self.model, self.data)
        self.base_controller.reset()
        self.arm_controller.reset()

    def control_callback(self, *_):
        command = None if self.command_queue.empty() else self.command_queue.get()

        if isinstance(command, tuple) and command[0] == "set_seed":
            self.set_seed(command[1])

        elif command == 'reset':
            self.reset()

        self.base_controller.control_callback(command)
        self.arm_controller.control_callback(command)
        self.update_shm_state()


class MujocoEnv(CommonMujocoEnv):
    def __init__(self, cfg: MujocoEnvConfig, render_images=True, show_viewer=True, show_images=False):
        super().__init__(cfg, render_images, show_viewer, show_images)

        self.physics_proc = mp.Process(target=self.physics_loop, daemon=True)
        self.physics_proc.start()

    def physics_loop(self):
        sim = YamBaseArmSim(
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
