import argparse
import os
import sys
import time
import numpy as np
import torch
import pyrallis

from itertools import count
from scipy.spatial.transform import Rotation as R

from constants import POLICY_CONTROL_PERIOD
from envs.common_mj_env import MujocoEnvConfig
from envs.utils.camera_utils import pcl_from_obs
from interactive_scripts.dataset_recorder import ActMode
from common_utils.eval_utils import check_for_interrupt
import common_utils
import open3d as o3d

from scripts.train_waypoint import load_waypoint
from scripts.train_dense import load_model

def run_dense_mode(env, dense_policy, dense_dataset, recorder, prev_obs):
    cached_actions = []
    mode = ActMode.Dense.value

    consecutive_modes_required = 5
    WAYPOINT_THRESH = 0.5
    TERMINATE_THRESH = 1.3
    mode_history = []

    for step_idx in count():
        loop_start = time.time()

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
        raw_mode = raw_mode.item()

        mode_history.append(raw_mode)
        if len(mode_history) >= consecutive_modes_required:
            if np.all(np.array(mode_history) < WAYPOINT_THRESH):
                return ActMode.ArmWaypoint.value if env.cfg.wbc else ActMode.BaseWaypoint.value, prev_obs
            elif np.all(np.array(mode_history) > TERMINATE_THRESH):
                return ActMode.Terminate.value, prev_obs
            mode_history = []

        ee_pos = prev_obs["arm_pos"]
        ee_quat = prev_obs["arm_quat"]
        base_pose = prev_obs["base_pose"]

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

        if recorder:
            recorder.add_numpy(obs, ["viewer_image"], color=(255, 140, 0))

        if check_for_interrupt() or env.num_step > env.max_num_step or (env.num_step > 50 and obs["reward"]):
            return ActMode.Terminate.value, obs

        # Enforce control frequency
        elapsed = time.time() - loop_start
        sleep_time = POLICY_CONTROL_PERIOD - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

        prev_obs = obs  # slide window

    return mode, prev_obs  # unreachable, but keeps the signature

def run_waypoint_mode(env, waypoint_policy, num_pass, recorder, curr_mode):
    obs = env.get_obs()

    while curr_mode in [ActMode.ArmWaypoint.value, ActMode.BaseWaypoint.value]:
        if recorder:
            recorder.add_numpy(obs, ["viewer_image"])

        points, colors = pcl_from_obs(obs, env.cfg)
        proprio = obs["proprio"]
        if len(points) == 0:
            print(f"Terminating episode because point cloud is empty")
            return ActMode.Terminate.value, obs

        with torch.no_grad():
            _, pos_cmd, rot_cmd, gripper_cmd, next_mode = waypoint_policy.inference(
                torch.from_numpy(points).float(),
                torch.from_numpy(colors).float(),
                torch.from_numpy(proprio).float(),
                num_pass=num_pass,
            )
            print(obs['arm_pos'], pos_cmd)

            if waypoint_policy.cfg.use_euler:
                quat_cmd = R.from_euler("xyz", rot_cmd).as_quat()
            else:
                quat_cmd = rot_cmd

        if curr_mode == ActMode.ArmWaypoint.value:
            if quat_cmd[3] < 0:
                np.negative(quat_cmd, out=quat_cmd)
            reached, err = env.move_to_arm_waypoint(pos_cmd, quat_cmd, gripper_cmd, recorder=recorder)
        else:
            base_pose = np.array([pos_cmd[0], pos_cmd[1], R.from_quat(quat_cmd).as_euler("xyz")[2]])
            reached, err = env.move_to_base_waypoint(base_pose, recorder=recorder)
        curr_mode = next_mode

        obs = env.get_obs()

        if check_for_interrupt() or env.num_step > env.max_num_step or obs["reward"]:
            return ActMode.Terminate.value, obs

    return curr_mode, obs


def eval_hybrid(
    waypoint_policy,
    dense_policy,
    dense_dataset,
    env,
    seed,
    num_pass,
    save_dir,
    record,
):
    assert not waypoint_policy.training
    assert not dense_policy.training

    env.seed(seed)
    env.reset()

    recorder = None
    if record:
        recorder = common_utils.Recorder(save_dir)

    mode = ActMode.ArmWaypoint.value if env.cfg.wbc else ActMode.BaseWaypoint.value
    obs = env.get_obs()

    while mode != ActMode.Terminate.value:
        if mode in [ActMode.ArmWaypoint.value, ActMode.BaseWaypoint.value]:
            mode, obs = run_waypoint_mode(env, waypoint_policy, num_pass, recorder, mode)
        elif mode == ActMode.Dense.value:
            mode, obs = run_dense_mode(env, dense_policy, dense_dataset, recorder, obs)

    if recorder:
        recorder.save(f"s{seed}", fps=10)

    return obs["reward"], env.num_step

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--waypoint_model", type=str, required=True)
    parser.add_argument("-d", "--dense_model", type=str, required=True)
    parser.add_argument("--seed", type=int, default=99999)
    parser.add_argument("--num_episode", type=int, default=10)
    parser.add_argument("--num_pass", type=int, default=3)
    parser.add_argument("--save_dir", type=str, default="rollouts")
    parser.add_argument("--record", type=int, default=1)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("-e", "--env_cfg", type=str, required=True)
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

    waypoint_policy = load_waypoint(args.waypoint_model, device="cuda").cuda()
    waypoint_policy.eval()

    dense_policy, dense_dataset, _ = load_model(args.dense_model, "cuda", load_only_one=True)
    dense_policy.eval()

    if dense_policy.cfg.use_ddpm:
        common_utils.cprint(f"Warning: overriding to use ddim with 10 steps")
        dense_policy.cfg.use_ddpm = 0
        dense_policy.cfg.ddim.num_inference_timesteps = 10

    idx = 0
    if os.path.exists(args.save_dir):
        idx = len([fn for fn in os.listdir(args.save_dir) if "mp4" in fn])
    args.seed += idx

    if args.save_dir:
        log_path = os.path.join(args.save_dir, "eval.log")
        if os.path.exists(log_path):
            sys.stdout = common_utils.Logger(log_path, mode="a", print_to_stdout=True)
        else:
            sys.stdout = common_utils.Logger(log_path, mode="w", print_to_stdout=True)

    scores = []
    for idx, seed in enumerate(range(args.seed, args.seed + args.num_episode)):
        env = MujocoEnv(env_cfg, show_viewer=not args.headless, show_images=False)
        score, num_step = eval_hybrid(
            waypoint_policy,
            dense_policy,
            dense_dataset,
            env,
            seed,
            num_pass=args.num_pass,
            save_dir=args.save_dir,
            record=args.record,
        )
        scores.append(score)
        print(f"[{idx+1}/{args.num_episode}], % Success: {np.mean(scores):.4f}, # Success: {int(np.sum(scores))}")
        print(common_utils.wrap_ruler("", max_len=80))
        env.close()


if __name__ == "__main__":
    # python scripts/eval_hybrid.py -w exps/waypoint/cabinet_wbc/latest.pt -d exps/dense/cabinet_wbc_delta_allcams/latest.pt --num_episode 20 -e envs/cfgs/open_wbc.yaml

    # python scripts/eval_hybrid.py -w exps/waypoint/cabinet_base_arm/latest.pt -d exps/dense/cabinet_base_arm_delta_allcams/latest.pt --num_episode 20 -e envs/cfgs/open_base_arm.yaml

    # python scripts/eval_hybrid.py -w exps/waypoint/dishwasher_wbc/latest.pt -d exps/dense/dishwasher_wbc_delta_allcams/latest.pt --num_episode 20 -e envs/cfgs/dishwasher_wbc.yaml

    # python scripts/eval_hybrid.py -w exps/waypoint/dishwasher_base_arm/latest.pt -d exps/dense/dishwasher_base_arm_delta_allcams/latest.pt --num_episode 20 -e envs/cfgs/dishwasher_base_arm.yaml

    # python scripts/eval_hybrid.py -w exps/waypoint/cube_base_arm/latest.pt -d exps/dense/cube_base_arm_delta/latest.pt --num_episode 20 -e envs/cfgs/cube_base_arm.yaml

    main()
