from dataclasses import dataclass
from collections import defaultdict, namedtuple
import os
import numpy as np
import torch
import torchvision.transforms as transforms

from common_utils import get_all_files
from interactive_scripts.dataset_recorder import ActMode, load_episode
from scipy.spatial.transform import Rotation as R

class DenseInputProcessor:
    def __init__(self, camera_names: list[str], target_size: int):
        self.camera_names = camera_names
        self.target_size = target_size
        self.rescale_transform = transforms.Resize(
            (target_size, target_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,  # type: ignore
        )

    def process(self, obs: dict):
        processed_obs = {}
        for k, v in obs.items():
            if k == "proprio":
                processed_obs["prop"] = torch.from_numpy(v.astype(np.float32))

            if k not in self.camera_names:
                continue

            v = torch.from_numpy(v.copy())
            v = v.permute(2, 0, 1)
            v = self.rescale_transform(v)
            processed_obs[k] = v
        if not "prop" in processed_obs:
            proprio = np.hstack((obs["arm_pos"], obs["arm_quat"], obs["gripper_pos"], obs["base_pose"]))
            processed_obs["prop"] = torch.from_numpy(proprio.astype(np.float32))
        return processed_obs


Batch = namedtuple("Batch", ["obs", "action"])

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATASETS = {
    "pillow_base_arm": os.path.join(PROJECT_ROOT, "data/dev_pillow_base_arm"),
}

@dataclass
class DenseDatasetConfig:
    path: str = ""
    camera_views: str = "wrist_view"
    image_size: int = 96
    use_interpolate: int = 1
    wbc: int = 1
    delta_actions: int = 0
    predict_mode: int = 1
    num_data: int = -1

    def __post_init__(self):
        if self.path in DATASETS:
            self.path = DATASETS[self.path]


class DenseDataset:
    def __init__(self, cfg: DenseDatasetConfig, load_only_one=False):
        self.cfg = cfg
        # load_only_one makes loading faster for non-training purpose
        self.load_only_one = load_only_one
        self.camera_views = cfg.camera_views.split("+")
        self.input_processor = DenseInputProcessor(self.camera_views, cfg.image_size)

        self.episodes: list[list[dict]] = self._load_and_process_episodes(cfg.path, cfg.num_data)
        self.idx2entry = {}  # map from a single number to
        for episode_idx, episode in enumerate(self.episodes):
            for step_idx in range(len(episode)):
                self.idx2entry[len(self.idx2entry)] = (episode_idx, step_idx)

        print(f"Dataset loaded from {cfg.path}")
        print(f"  episodes: {len(self.episodes)}")
        print(f"  steps: {len(self.idx2entry)}")
        print(f"  avg episode len: {len(self.idx2entry) / len(self.episodes):.1f}")

    @property
    def action_dim(self) -> int:
        return self.episodes[0][0]["action"].size(0)

    @property
    def obs_shape(self) -> tuple[int]:
        return self.episodes[0][0][self.camera_views[0]].size()

    @property
    def prop_dim(self) -> int:
        return self.episodes[0][0]["prop"].size(0)

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        return self.episodes[idx]

    def process_observation(self, obs):
        return self.input_processor.process(obs)

    def _load_and_process_episodes(self, path, num_data):
        print(f"loading data from {path}")
        pkl_files = list(sorted(get_all_files(path, "pkl")))
        if self.load_only_one:
            pkl_files = pkl_files[:1]

        all_episodes: list[list[dict]] = []

        # Label the last 10 timesteps of the demo as terminal, to avoid label imbalance
        TERMINATE_WINDOW = 10

        for episode_idx, f in enumerate(sorted(pkl_files)):
            if num_data > 0 and episode_idx >= num_data:
                break

            success_msg = ""

            raw_episode = load_episode(f)

            episode = []

            for t, timestep in enumerate(raw_episode):
                if self.cfg.use_interpolate:
                    if timestep["mode"] in [ActMode.ArmWaypoint, ActMode.BaseWaypoint]: # Use dense and interpolation data
                        continue
                else:
                    if timestep["mode"] != ActMode.Dense: # Use only strictly dense data
                        continue

                # The action consists of: 3 dims for pos, 4 for rot, 1 for gripper, 2 for base, 1 for mode (Waypoint/Dense/Terminate)
                arm_action = np.zeros(8) # 3 pos, 4 quat, 1 gripper
                base_action = np.zeros(3) # 2 xy pos, 1 theta
                next_mode = None

                ee_pos = timestep["action"][:3]
                arm_action[7] = timestep["action"][7] 
                base_pose = timestep["action"][8:11]

                ee_quat = R.from_euler('xyz', timestep["action"][3:6]).as_quat()
                if ee_quat[0] < 0:
                    ee_quat *= -1

                delta_ee_pos = timestep["delta_action"][:3]
                delta_ee_quat = timestep["delta_action"][3:7]
                delta_base_pose = timestep["delta_action"][8:11]

                if self.cfg.delta_actions:
                    arm_action[:3] = delta_ee_pos 
                    arm_action[3:7] = delta_ee_quat
                    base_action = delta_base_pose 
                else:
                    arm_action[:3] = ee_pos 
                    arm_action[3:7] = ee_quat
                    base_action = base_pose 

                if t > len(raw_episode) - TERMINATE_WINDOW:
                    next_mode = ActMode.Terminate.value

                elif timestep["mode"] == ActMode.Interpolate:
                    for k in range(t + 1, len(list(raw_episode))):
                        if raw_episode[k]["mode"] != ActMode.Interpolate:
                            next_mode = raw_episode[k]["mode"].value

                elif timestep["mode"] == ActMode.Dense:
                    next_mode = raw_episode[t + 1]["mode"].value

                if self.cfg.wbc:
                    action = arm_action
                else:
                    action = np.hstack((arm_action, base_action))
                    
                if self.cfg.predict_mode:
                    action = np.hstack((action, next_mode))

                processed_timestep = {
                    "is_dense": torch.tensor(float(timestep["mode"] == ActMode.Dense)),
                    "action": torch.from_numpy(action).float(),
                }
                processed_timestep.update(self.process_observation(timestep["obs"]))
                episode.append(processed_timestep)

                if not success_msg and timestep.get("reward", 0) > 0:
                    success_msg = f", success since {len(episode)}"

            print(f"episode {episode_idx}, len: {len(episode)}" + success_msg)
            all_episodes.append(episode)

        return all_episodes

    def get_action_range(self) -> tuple[torch.Tensor, torch.Tensor]:

        action_max = self.episodes[0][0]["action"]
        action_min = self.episodes[0][0]["action"]

        for episode in self.episodes:
            for timestep in episode:
                action_max = torch.maximum(action_max, timestep["action"])
                action_min = torch.minimum(action_min, timestep["action"])

        print(f"raw action value range, the model should do all the normalization:")
        for i in range(len(action_min)):
            print(f"  dim {i}, min: {action_min[i].item():.5f}, max: {action_max[i].item():.5f}")

        return action_min, action_max

    def _convert_to_batch(self, samples, device):
        batch = {}
        for k, v in samples.items():
            batch[k] = torch.stack(v).to(device)

        action = {"action": batch.pop("action")}
        ret = Batch(obs=batch, action=action)
        return ret

    def _stack_actions(self, idx, begin, action_len):
        """stack actions in [begin, end)"""
        episode_idx, step_idx = self.idx2entry[idx]
        episode = self.episodes[episode_idx]

        actions = []
        valid_actions = []
        for action_idx in range(begin, begin + action_len):
            if action_idx < 0:
                actions.append(torch.zeros_like(episode[step_idx]["action"]))
                valid_actions.append(0)
            elif action_idx < len(episode):
                actions.append(episode[action_idx]["action"])
                valid_actions.append(1)
            else:
                actions.append(torch.zeros_like(actions[-1]))
                valid_actions.append(0)

        valid_actions = torch.tensor(valid_actions, dtype=torch.float32)
        actions = torch.stack(actions, dim=0)
        return actions, valid_actions

    def sample_dp(self, batchsize, action_pred_horizon, device):
        indices = np.random.choice(len(self.idx2entry), batchsize)
        samples = defaultdict(list)
        for idx in indices:
            episode_idx, step_idx = self.idx2entry[idx]
            entry: dict = self.episodes[episode_idx][step_idx]

            actions, valid_actions = self._stack_actions(idx, step_idx, action_pred_horizon)
            assert torch.equal(actions[0], entry["action"])

            samples["valid_action"].append(valid_actions)
            for k, v in entry.items():
                if k == "action":
                    samples[k].append(actions)
                else:
                    samples[k].append(v)

        return self._convert_to_batch(samples, device)


def visualize_episode(episode, image_size, camera):
    from common_utils import generate_grid, plot_images, RandomAug
    import os

    aug = RandomAug(pad=6)

    is_dense = []
    action_dims = [[] for _ in range(8)]

    for timestep in episode:
        action = timestep["action"]
        is_dense.append(timestep["is_dense"])
        for i, adim_val in enumerate(action):
            action_dims[i].append(adim_val.item())

    fig, axes = generate_grid(cols=8, rows=1)
    for idx, adim_vals in enumerate(action_dims):
        axes[idx].plot(adim_vals)
        axes[idx].set_title(f"action dim {idx}")

        for i, dense in enumerate(is_dense):
            if dense > 0:
                axes[idx].axvspan(i, i + 1, facecolor="green", alpha=0.3, label="dense")

        axes[idx].set_xlim(0, len(is_dense))

    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(__file__), "actions.png"))

    images = [obs[camera] for obs in episode]
    images = images[::8]
    images = aug(torch.stack(images).float())
    images = [img.permute(1, 2, 0).numpy().astype(int) for img in images]
    fig = plot_images(images)
    path = os.path.join(os.path.dirname(__file__), "observations.png")
    print(f"saving image to {path}")
    fig.savefig(path)


def test():
    cfg = DenseDatasetConfig(
        path="data/dev_pillow_base_arm",
        camera_views="base1_image+wrist_image",
        use_interpolate=1,
        wbc=0,
        delta_actions=0,
        image_size=96,
    )
    dataset = DenseDataset(cfg, load_only_one=True)
    dataset.get_action_range()
    #visualize_episode(dataset.episodes[0], cfg.image_size, "base1_image")

if __name__ == "__main__":
    test()
