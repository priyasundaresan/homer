import pyrallis
import random
import argparse
import multiprocessing as mp
import time
from threading import Thread
import mujoco
import numpy as np
from common_utils.eval_utils import check_for_interrupt
from scipy.spatial.transform import Rotation as R
from ruckig import Result
from constants import POLICY_CONTROL_PERIOD
from interactive_scripts.dataset_recorder import ActMode
from teleop.policies import TeleopPolicy
from envs.utils.wbc_ik_solver import IKSolver
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

class MujocoSim(CommonMujocoSim):
    def __init__(self, task, mjcf_path, command_queue, shm_state, cfg, show_viewer=True):
        super().__init__(task, mjcf_path, command_queue, shm_state, show_viewer)
        self.cfg = cfg

        self.wbc_ik_solver = IKSolver(
            reset_qpos=self.cfg.arm_reset_qpos,
            base_immobile=self.cfg.base_immobile,
            collision_avoidance=self.cfg.collision_avoidance,
        )

        # Cache references to array slices
        self.arm_dofs = arm_dofs = 7
        self.qpos_arm = self.data.qpos[self.base_dofs:(self.base_dofs + self.arm_dofs)]
        qpos_arm = self.data.qpos[self.base_dofs:(self.base_dofs + self.arm_dofs)]
        qvel_arm = self.data.qvel[self.base_dofs:(self.base_dofs + self.arm_dofs)]
        ctrl_arm = self.data.ctrl[self.base_dofs:(self.base_dofs + self.arm_dofs)]
        self.qpos_gripper = self.data.qpos[(self.base_dofs + self.arm_dofs):(self.base_dofs + self.arm_dofs + 1)]
        ctrl_gripper = self.data.ctrl[(self.base_dofs + self.arm_dofs):(self.base_dofs + self.arm_dofs + 1)]
        self.arm_controller = ArmController(qpos_arm, qvel_arm, ctrl_arm, self.qpos_gripper, ctrl_gripper, self.model.opt.timestep, self.cfg.arm_reset_qpos, wbc=True)

        # Reset the environment
        self.reset()

        # Set control callback
        mujoco.set_mjcb_control(self.control_callback)

    def reset(self):
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)

        self.reset_task()
        mujoco.mj_forward(self.model, self.data)

        # Reset controllers
        self.base_controller.reset()
        self.arm_controller.reset()

        self.wbc_ik_solver.configuration.update(self.data.qpos[:self.base_dofs+self.arm_dofs+8])

    def control_callback(self, *_):
        # Check for new command
        command = None if self.command_queue.empty() else self.command_queue.get()

        if isinstance(command, tuple) and command[0] == "set_seed":
            self.set_seed(command[1])

        if command == 'reset':
            self.reset()

        if command is not None and 'arm_pos' in command:
            full_qpos = self.wbc_ik_solver.solve(
                command['arm_pos'], command['arm_quat'], np.hstack([self.qpos_base, self.qpos_arm, np.zeros(8)])
            )

            # Distribute to base and arm controllers
            command['base_pose'] = full_qpos[:3]
            command['arm_qpos'] = full_qpos[3:10]

        self.base_controller.control_callback(command)
        self.arm_controller.control_callback(command)
       
        self.update_shm_state()

class MujocoEnv(CommonMujocoEnv):
    def __init__(self, cfg: MujocoEnvConfig, render_images=True, show_viewer=True, show_images=False):
        super().__init__(cfg, render_images, show_viewer, show_images)

        self.physics_proc = mp.Process(target=self.physics_loop, daemon=True)
        self.physics_proc.start()

    def physics_loop(self):
        sim = MujocoSim(self.task, self.mjcf_path, self.command_queue, self.shm_state, self.cfg, show_viewer=self.show_viewer)

        if self.render_images:
            Thread(target=self.render_loop, args=(sim.model, sim.data), daemon=True).start()

        sim.launch()

    def close(self):
        super().close()
        if self.physics_proc is not None and self.physics_proc.is_alive():
            self.physics_proc.terminate()
            self.physics_proc.join()
            self.physics_proc = None  # Clear reference

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_cfg", type=str, default="envs/cfgs/cube_wbc_distractor.yaml")
    #parser.add_argument("--env_cfg", type=str, default="envs/cfgs/open_wbc.yaml")
    args = parser.parse_args()

    env_cfg = pyrallis.load(MujocoEnvConfig, open(args.env_cfg, "r"))
    env = MujocoEnv(env_cfg)

    # Test close
    #env.close()

    ## Test WBC waypoint reaching
    while True:
        env.reset()
        time.sleep(0.5)
        target_pos = np.random.uniform(low=(-1.0, -0.3, 0.3), high=(2.0, 0.3, 0.7))
        target_quat = np.random.uniform(-0.8, 0.8, 4) + [np.sqrt(2)/2, np.sqrt(2)/2, 0.0, 0.0]
        target_quat = target_quat/np.linalg.norm(target_quat)
        gripper_pos = random.choice((0,1))
        reached, pos_error = env.move_to_arm_waypoint(target_pos, target_quat, gripper_pos)
        print(pos_error)

    #while True:
    #    env.reset()
    #    random_pos = np.random.uniform(low=(0.5, -0.3, 0.1), high=(0.9, 0.3, 0.9))
    #    random_quat = np.random.uniform(-0.4, 0.4, 4) + [np.sqrt(2)/2, np.sqrt(2)/2, 0.0, 0.0]
    #    random_quat = random_quat/np.linalg.norm(random_quat)

    #    print(random_pos)
    #    for _ in range(100):
    #        action = {
    #            'base_pose': np.zeros(3), # No base pos, handled by WBC
    #            'arm_pos': random_pos, 
    #            'arm_quat': random_quat,
    #            'gripper_pos': np.random.rand(1)
    #        }
    #        env.step(action)
    #        obs = env.get_obs()
    #        print(action['arm_pos'], obs['arm_pos'])
    #        
    #        #print([(k, v.shape) if v.ndim == 3 else (k, v) for (k, v) in obs.items()])
    #        time.sleep(POLICY_CONTROL_PERIOD)  # Note: Not precise

    # Test data collection
    #env.collect_episode()

    # Test replay
    #env.replay_episode('dev1/demo00000.pkl', replay_mode='absolute')
    #env.replay_episode('dev1/demo00000.pkl', replay_mode='delta')
