import pyrallis
import argparse
from envs.common_mj_env import MujocoEnvConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--env_cfg", type=str, default="envs/cfgs/cube_wbc.yaml")
    #parser.add_argument("--env_cfg", type=str, default="envs/cfgs/cube_base_arm.yaml")
    #parser.add_argument("--env_cfg", type=str, default="envs/cfgs/open_base_arm.yaml")
    #parser.add_argument("--env_cfg", type=str, default="envs/cfgs/open_wbc.yaml")
    #parser.add_argument("--env_cfg", type=str, default="envs/cfgs/dishwasher_wbc.yaml")
    #parser.add_argument("--env_cfg", type=str, default="envs/cfgs/dishwasher_base_arm.yaml")

    args = parser.parse_args()
    env_cfg = pyrallis.load(MujocoEnvConfig, open(args.env_cfg, "r"))

    robot = getattr(env_cfg, 'robot', 'kinova')
    if robot == 'yam':
        if env_cfg.wbc:
            from envs.mj_env_wbc_yam import MujocoEnv
        else:
            from envs.mj_env_base_arm_yam import MujocoEnv
    else:  # kinova (default)
        if env_cfg.wbc:
            from envs.mj_env_wbc import MujocoEnv
        else:
            from envs.mj_env_base_arm import MujocoEnv
    
    #env = MujocoEnv(env_cfg, show_images=True) # LINUX
    env = MujocoEnv(env_cfg, show_images=False) # MAC
    env.reset()

    while True:
        env.collect_episode()
        print('episode length: %d, max steps: %d'%(env.num_step, env.max_num_step))
