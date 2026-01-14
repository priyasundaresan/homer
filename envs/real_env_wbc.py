from envs.common_real_env import CommonRealEnv, RealEnvConfig
from envs.utils.wbc_ik_solver import IKSolver
from scipy.spatial.transform import Rotation as R
import numpy as np
from constants import ARM_X_OFFSET, BASE_HEIGHT

class RealEnv(CommonRealEnv):
    def __init__(self, cfg: RealEnvConfig):
        super().__init__(cfg)
        assert(self.cfg.wbc)
        self.wbc_ik_solver = IKSolver(
            base_immobile=False,
            collision_avoidance=False,
            posture_cost=2e-3,
        )
        self.RESET_QPOS = np.array([0., 0., 0., 0., -0.34906585, 3.14159265, -2.54818071, 0., -0.87266463, 1.57079633, 0., 0., 0., 0., 0., 0., 0., 0.])
        self.arm_base_offset = [ARM_X_OFFSET, 0, BASE_HEIGHT] # arm is forward (0.1199m) and raised by base height (0.3948m)

    def reset(self):
        super().reset()
        self.wbc_ik_solver.configuration.update(self.RESET_QPOS)
        print('WBC IK Solver has been reset')

    def step(self, action):
        qpos_base = self.base.get_state()['base_pose']
        qpos_arm = self.arm.get_qpos()
        step_action = action.copy()

        if 'arm_pos' in action:
            T_action = np.eye(4)
            T_action[:3, :3] = R.from_euler('z', qpos_base[2]).as_matrix()
            T_action[:3, 3] = np.array([qpos_base[0], qpos_base[1], 0]) + self.arm_base_offset
            arm_pos_adjusted = T_action@np.array([step_action['arm_pos'][0], step_action['arm_pos'][1], step_action['arm_pos'][2], 1.0])
            arm_pos_adjusted = arm_pos_adjusted[:3]
            step_action['arm_pos'] = arm_pos_adjusted

            action_qpos = self.wbc_ik_solver.solve(step_action['arm_pos'], \
                                                   step_action['arm_quat'], \
                                                   np.hstack([qpos_base, qpos_arm, np.zeros(8)]))

            action_base_pose = action_qpos[:3]
            action_arm_qpos = action_qpos[3:10]
            action_arm_qpos = action_arm_qpos % (2 * np.pi) # Unwrapping
            step_action['base_pose'] = action_base_pose
            step_action['arm_qpos'] = action_arm_qpos

        #print(step_action['base_pose'].round(2), step_action['arm_qpos'].round(2))
        #[ 0. -0. -0.] [0.   5.94 3.14 3.74 0.   5.42 1.57]
        #[ 0. -0. -0.] [0.02 5.95 3.16 3.75 0.01 5.43 1.6 ]
        self.base.execute_action(step_action)  # Non-blocking
        self.arm.execute_action(step_action)   # Non-blocking

if __name__ == '__main__':
    import argparse
    import pyrallis
    import time
    from common_utils import Stopwatch
    from constants import POLICY_CONTROL_PERIOD

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_cfg", type=str, default="envs/cfgs/real_wbc.yaml")

    args = parser.parse_args()
    env_cfg = pyrallis.load(RealEnvConfig, open(args.env_cfg, "r"))

    try:
        env = RealEnv(env_cfg)
        env.reset()
    finally:
        env.close()
        print('Successfully closed env.')

    ## Test get obs
    #try:
    #    env = RealEnv(env_cfg)
    #    env.reset()
    #    stopwatch = Stopwatch()
    #    while True:
    #        with stopwatch.time('get_obs'):
    #            obs = env.get_obs()
    #        lat_ms = stopwatch.times['get_obs'][-1]
    #        bar_len = int(min(lat_ms / 2, 50))  # Scale: 2ms = 1 char, max 50 chars
    #        bar = '*' * bar_len
    #        print(f"[get_obs] {lat_ms:6.1f} ms | {bar}")
    #        time.sleep(POLICY_CONTROL_PERIOD)  # Note: Not precise
    #finally:
    #    env.close()
    #    print('Successfully closed env.')

    #try:
    #    env = RealEnv(env_cfg)
    #    env.collect_episode()
    #finally:
    #    env.close()
    #    print('Successfully closed env.')
