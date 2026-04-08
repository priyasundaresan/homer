import argparse
import copy
import numpy as np
import torch
import os
import pyrallis
import mujoco as mj

from envs.common_mj_env import MujocoEnvConfig
from scipy.spatial.transform import Rotation as R
import common_utils

import os
import sys
import time
from itertools import count
from constants import POLICY_CONTROL_PERIOD
from scripts.train_dense import load_model
from common_utils.eval_utils import (
    check_for_interrupt,
)
from contextlib import nullcontext

def eval_dense(
    dense_policy,
    dense_dataset,
    env,
    seed,
    save_dir,
    record,
):
    assert not dense_policy.training

    env.seed(seed)
    env.reset()

    recorder = None
    if record:
        assert save_dir is not None
        recorder = common_utils.Recorder(save_dir)

    cached_actions = []
    prev_obs = env.get_obs()  # This is obs_t

    for step_idx in count():
        loop_start = time.time()

        ee_pos = prev_obs['arm_pos']
        ee_quat = prev_obs['arm_quat']
        base_pose = prev_obs['base_pose']

        dense_obs = dense_dataset.process_observation(prev_obs)
        dense_obs = {k: v.cuda(non_blocking=True) for k, v in dense_obs.items()}

        if len(cached_actions) == 0:
            action_seq = dense_policy.act(dense_obs)
            for action in action_seq.split(1, dim=0):
                cached_actions.append(action.squeeze(0))
        action = cached_actions.pop(0)

        if not dense_dataset.cfg.wbc:
            action_pos, action_quat, action_gripper, action_base, raw_mode = action.split([3, 4, 1, 3, 1])
            action_base = action_base.detach().cpu().numpy()
        else:
            action_pos, action_quat, action_gripper, raw_mode = action.split([3, 4, 1, 1])
            action_base = np.zeros(3)

        action_pos = action_pos.detach().cpu().numpy()
        action_quat = action_quat.detach().cpu().numpy()
        action_gripper = action_gripper.detach().cpu().numpy()

        if dense_dataset.cfg.delta_actions:
            if not env.cfg.wbc:
                ee_pos = env.global_to_local_arm_pos(ee_pos, base_pose)
                action_base += base_pose

            action_pos += ee_pos
            action_rot = R.from_quat(action_quat) * R.from_quat(ee_quat)
            action_quat = action_rot.as_quat()

        action_quat /= np.linalg.norm(action_quat)
        if action_quat[3] < 0.0:
            np.negative(action_quat, out=action_quat)

        action_dict = {
            "base_pose": action_base,
            "arm_pos": action_pos,
            "arm_quat": action_quat,
            "gripper_pos": action_gripper,
        }

        env.step(action_dict)
        obs = env.get_obs()

        if recorder is not None:
            recorder.add_numpy(obs, ["viewer_image"], color=(255, 140, 0))

        if check_for_interrupt() or env.num_step > env.max_num_step or (env.num_step > 100 and obs['reward']):
            break

        prev_obs = obs

        elapsed = time.time() - loop_start
        sleep_time = POLICY_CONTROL_PERIOD - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    if recorder is not None:
        recorder.save(f"s{seed}", fps=10)

    return obs['reward'], env.num_step

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dense_model", type=str, required=True, help="dense model path")
    parser.add_argument("--seed", type=int, default=99999)
    parser.add_argument("--num_episode", type=int, default=20)
    parser.add_argument("--num_pass", type=int, default=3)
    parser.add_argument("--save_dir", type=str, default='rollouts')
    parser.add_argument("--record", type=int, default=1)
    parser.add_argument("--headless", action='store_true')
    parser.add_argument("-e", "--env_cfg", type=str, default="envs/cfgs/cube_base_arm.yaml")
    args = parser.parse_args()

    env_cfg = pyrallis.load(MujocoEnvConfig, open(args.env_cfg, "r"))
    robot = getattr(env_cfg, 'robot', 'kinova')
    if robot == 'yam':
        if env_cfg.wbc:
            from envs.mj_env_wbc_yam import MujocoEnv
        else:
            from envs.mj_env_base_arm_yam import MujocoEnv
    else:
        if env_cfg.wbc:
            from envs.mj_env_wbc import MujocoEnv
        else:
            from envs.mj_env_base_arm import MujocoEnv

    print(f">>>>>>>>>>{args.dense_model}<<<<<<<<<<")
    dense_policy, dense_dataset, _ = load_model(args.dense_model, "cuda", load_only_one=True)
    dense_policy.eval()

    if dense_policy.cfg.use_ddpm:
        common_utils.cprint(f"Warning: override to use ddim with step 10")
        dense_policy.cfg.use_ddpm = 0
        dense_policy.cfg.ddim.num_inference_timesteps = 10

    scores = []

    idx = 0
    if os.path.exists('rollouts'):
        idx = len([fn for fn in os.listdir('rollouts') if 'mp4' in fn])
    args.seed += idx

    if args.save_dir is not None:
        log_path = os.path.join(args.save_dir, "eval.log")
        if os.path.exists(log_path):
            sys.stdout = common_utils.Logger(log_path, mode="a", print_to_stdout=True)
        else:
            sys.stdout = common_utils.Logger(log_path, mode="w", print_to_stdout=True)

    for idx, seed in enumerate(range(args.seed, args.seed + args.num_episode)):
        env = MujocoEnv(env_cfg, show_viewer=not args.headless, show_images=False)

        score, num_step = eval_dense(
            dense_policy,
            dense_dataset,
            env,
            seed=seed,
            save_dir=args.save_dir,
            record=args.record,
        )
        scores.append(score)
        print(f"[{idx+1}/{args.num_episode}], % Success: {np.mean(scores):.4f}, # Success: {int(np.sum(scores))}")
        print(common_utils.wrap_ruler("", max_len=80))

        env.close()

if __name__ == "__main__":
    ### Example Usage
    
    # --- Local evaluation (with display) ---
    # python scripts/eval_dense.py -d exps/dense/cube_wbc_delta/latest.pt -e envs/cfgs/cube_wbc.yaml
    # python scripts/eval_dense.py -d exps/dense/cube_base_arm_delta/latest.pt -e envs/cfgs/cube_base_arm.yaml
    # python scripts/eval_dense.py -d exps/dense/cabinet_wbc_delta_allcams/latest.pt -e envs/cfgs/open_wbc.yaml
    # python scripts/eval_dense.py -d exps/dense/cabinet_base_arm_delta_allcams/latest.pt -e envs/cfgs/open_base_arm.yaml
    # python scripts/eval_dense.py -d exps/dense/dishwasher_wbc_delta_allcams/latest.pt -e envs/cfgs/dishwasher_wbc.yaml
    # python scripts/eval_dense.py -d exps/dense/dishwasher_base_arm_delta_allcams/latest.pt -e envs/cfgs/dishwasher_base_arm.yaml
    
    # --- Headless evaluation (e.g., on a cluster) ---
    # python scripts/eval_dense.py ... --headless
    # If headless rendering fails, try:
    # MUJOCO_GL=egl python scripts/eval_dense.py ... --headless

    main()
