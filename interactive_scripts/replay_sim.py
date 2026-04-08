import pyrallis
import os
import argparse
from envs.common_mj_env import MujocoEnvConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="dev1")
    parser.add_argument("--mode", type=str, default="absolute")
    args = parser.parse_args()
    assert(args.mode in ["absolute", "delta"])
    data_dir = args.data_dir
    env_cfg_path = os.path.join(data_dir, "env_cfg.yaml")
    env_cfg = pyrallis.load(MujocoEnvConfig, open(env_cfg_path, "r"))

    if env_cfg.wbc:
        from envs.mj_env_wbc import MujocoEnv
    else:
        from envs.mj_env_base_arm import MujocoEnv

    env = MujocoEnv(env_cfg)
    for fn in sorted(os.listdir(data_dir)):
        if fn.endswith('.pkl'):
            env.replay_episode(os.path.join(data_dir, fn), replay_mode=args.mode)
