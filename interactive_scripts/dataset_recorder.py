import cv2
import numpy as np
from enum import Enum
import os
import glob
import pickle
from typing import Optional
import threading


class ActMode(Enum):
    ArmWaypoint = 0
    Dense = 1
    Terminate = 2
    BaseWaypoint = 3
    Interpolate = 4


def save_episode(episode, path):
    """Save episode as pickle (path should end in .pkl)."""
    with open(path, "wb") as f:
        pickle.dump(episode, f)


def load_episode(fn):
    """Load a .pkl episode file as a list of timestep dicts."""
    with open(fn, "rb") as f:
        return pickle.load(f)


class DatasetRecorder:
    def __init__(self, data_folder, vis_dim=(320, 240)):
        self.data_folder = data_folder
        os.makedirs(self.data_folder, exist_ok=True)
        self.vis_dim = vis_dim
        self.episode_idx = self.get_next_idx()

        self.flush_thread = None
        self._reset()

    def _reset(self):
        self.episode = []
        self.images = []
        self.waypoint_idx = -1

    def get_next_idx(self):
        existing_demos = glob.glob(os.path.join(self.data_folder, "demo*.pkl"))
        if len(existing_demos) == 0:
            return 0
        existing_indices = [
            int(os.path.basename(fname).split(".")[0][len("demo"):])
            for fname in existing_demos
        ]
        return np.max(existing_indices) + 1

    def record(
        self,
        mode: ActMode,
        obs: dict[str, np.ndarray],
        action: np.ndarray,
        delta_action: np.ndarray,
        teleop_mode: str,
        click_pos: Optional[np.ndarray] = None,
        reward: Optional[float] = None
    ):
        if mode in [ActMode.ArmWaypoint, ActMode.BaseWaypoint, ActMode.Interpolate]:
            print("Recording Click:", action)
            if mode != ActMode.Interpolate:
                self.waypoint_idx += 1
            waypoint_idx = self.waypoint_idx
        elif mode == ActMode.Dense:
            waypoint_idx = -1

        data = {
            "obs": obs,
            "action": action,
            "delta_action": delta_action,
            "mode": mode,
            "teleop_mode": teleop_mode,
            "waypoint_idx": waypoint_idx,
            "click": click_pos,
        }
        if reward is not None:
            data["reward"] = reward

        self.episode.append(data)

        views = [
            v for k, v in obs.items()
            if ("image" in k) and v.ndim == 3
        ]
        # Convert base_depth to RGB and append
        if "base1_depth" in obs:
            depth = obs["base1_depth"]  # shape (480, 640, 1)
            depth = np.squeeze(depth)  # shape (480, 640)

            # Normalize to 0-255 and convert to uint8
            depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
            depth_uint8 = depth_norm.astype(np.uint8)

            # Apply a colormap to make it RGB
            depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)

            views.append(depth_colored)
        if views:
            self.images.append(views)

    def _flush(self, episode, images, idx, save, visualize):
        if save and len(episode) > 0:
            mp4_path = os.path.join(self.data_folder, f"demo{idx:05d}.mp4")
            demo_path = os.path.join(self.data_folder, f"demo{idx:05d}.pkl")
            print(f"Saving to {demo_path}...")

            if images:
                vis_frames = []
                for i in range(len(images)):
                    image_list = images[i]  # list of raw images for this step
                    resized = [cv2.resize(img, self.vis_dim) for img in image_list]
                    stacked = np.hstack(resized)
                    if episode[i]["mode"] == ActMode.Dense:
                        stacked[:10, :, :] = (0, 255, 0)
                    vis_frames.append(cv2.cvtColor(stacked, cv2.COLOR_RGB2BGR))

                H, W, _ = vis_frames[0].shape
                out = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*"avc1"), 12, (W, H))
                for frame in vis_frames:
                    out.write(frame)
                out.release()

            save_episode(episode, demo_path)

            print(f"Finished saving demo{idx:05d}")
        else:
            print("Episode discarded")

    def end_episode(self, save, visualize=True):
        # snapshot episode and images before reset
        episode_copy = self.episode.copy()
        images_copy = self.images.copy()
        idx_copy = self.get_next_idx()

        self._reset()

        if save and episode_copy:
            self.episode_idx += 1
            print("Starting async save...")
            self.flush_thread = threading.Thread(
                target=self._flush,
                args=(episode_copy, images_copy, idx_copy, save, visualize),
                daemon=True
            )
            self.flush_thread.start()
        else:
            print("Episode not saved (empty or save=False)")

    def wait_for_flush(self):
        if self.flush_thread is not None:
            print("Waiting for async flush to finish...")
            self.flush_thread.join()
            self.flush_thread = None
