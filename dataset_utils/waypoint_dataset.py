import os
import pyrallis
from dataclasses import dataclass
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R

import common_utils
from interactive_scripts.dataset_recorder import ActMode, load_episode
from models.pointnet2_utils import farthest_point_sample
from models.triton_fps import triton_farthest_point_sample
from envs.common_mj_env import MujocoEnvConfig
from envs.utils.camera_utils import pcl_from_obs
from constants import BASE_HEIGHT

def augment_with_translation(
    points: torch.Tensor,
    colors: torch.Tensor,
    action_pos: torch.Tensor,
    proprio: torch.Tensor,
):
    loc_noise = torch.zeros(3)
    loc_noise.uniform_(-0.05, 0.05)
    aug_points = points + loc_noise.unsqueeze(0)

    color_noise = torch.zeros_like(colors)
    color_noise.normal_(0, 0.03)
    aug_colors = colors + color_noise

    assert action_pos.size() == loc_noise.size()
    aug_action_pos = action_pos + loc_noise
    aug_proprio = proprio.clone()
    aug_proprio[:3] += loc_noise

    return aug_points, aug_colors, aug_action_pos, aug_proprio

def augment_with_clusters(
    points: torch.Tensor,
    colors: torch.Tensor,
    user_clicked_labels: torch.Tensor = None,
    dist_to_click: torch.Tensor = None,
    click_point: torch.Tensor = None,
):
    """
    Adds random colored clusters around the mean of the point cloud.

    Args:
        points (torch.Tensor): Input point cloud of shape (N, 3).
        colors (torch.Tensor): Input colors of shape (N, 3).
        user_clicked_labels (torch.Tensor, optional): Binary or scalar label for each point. Shape (N,)
        dist_to_click (torch.Tensor, optional): Per-point distance to click. Shape (N,)
        click_point (torch.Tensor, optional): The 3D click position to recompute distances for new points.

    Returns:
        aug_points, aug_colors, aug_labels, aug_dist_to_click
    """
    N = points.shape[0]
    centroid = points.mean(dim=0)

    num_clusters = random.randint(2, 6)
    cluster_points = []
    cluster_colors = []

    for _ in range(num_clusters):
        points_per_cluster = random.randint(100, 800)
        cluster_radius = random.uniform(0.01, 0.04)
        center_offset = torch.empty(3).uniform_(-0.4, 0.4)
        cluster_center = centroid + center_offset
        offsets = torch.randn(points_per_cluster, 3) * cluster_radius
        cluster = cluster_center.unsqueeze(0) + offsets
        color = torch.rand(3).unsqueeze(0).expand(points_per_cluster, -1)

        cluster_points.append(cluster)
        cluster_colors.append(color)

    cluster_points = torch.cat(cluster_points, dim=0)
    cluster_colors = torch.cat(cluster_colors, dim=0)

    aug_points = torch.cat([points, cluster_points], dim=0)
    aug_colors = torch.cat([colors, cluster_colors], dim=0)

    aug_labels = None
    aug_dist = None

    if user_clicked_labels is not None:
        new_labels = torch.zeros(cluster_points.shape[0], dtype=user_clicked_labels.dtype)
        aug_labels = torch.cat([user_clicked_labels, new_labels], dim=0)

    if dist_to_click is not None and click_point is not None:
        new_dist = torch.norm(cluster_points - click_point.unsqueeze(0), dim=1)
        aug_dist = torch.cat([dist_to_click, new_dist], dim=0)

    return aug_points, aug_colors, aug_labels, aug_dist

def augment_with_rotation(
    points: torch.Tensor,
    action_pos: torch.Tensor,
    action_quat: torch.Tensor,
    proprio: torch.Tensor,
    rotate_scale: float,
):
    assert rotate_scale < 0.5
    random_rot = R.from_euler(
        "z", np.random.uniform(-np.pi * rotate_scale, np.pi * rotate_scale)
    ).as_matrix()
    random_rot = torch.from_numpy(random_rot).float()

    # points: [#point, 3]
    mu = points.mean(dim=0)
    aug_points = (points - mu) @ random_rot.T + mu
    aug_action_pos = (action_pos - mu) @ random_rot.T + mu

    aug_action_rot_mat = random_rot @ R.from_quat(action_quat.numpy()).as_matrix()
    aug_action_quat = R.from_matrix(aug_action_rot_mat).as_quat()
    if aug_action_quat[3] < 0:
        np.negative(aug_action_quat, out=aug_action_quat)
    aug_action_quat = torch.from_numpy(aug_action_quat).float()

    assert proprio.size(0) == 11, "proprio must be pos(3), quat(4), gripper(1), base(3)"
    aug_ee_pos = (proprio[:3] - mu) @ random_rot.T + mu

    aug_ee_quat = R.from_matrix(
        random_rot @ R.from_quat(proprio[3:7].numpy()).as_matrix()
    ).as_quat()
    if aug_ee_quat[3] < 0:
        np.negative(aug_ee_quat, out=aug_ee_quat)
    aug_ee_quat = torch.from_numpy(aug_ee_quat).float()

    aug_proprio = torch.cat((aug_ee_pos, aug_ee_quat, proprio[7:]))
    assert aug_proprio.size() == proprio.size()

    return aug_points, aug_action_pos, aug_action_quat, aug_proprio

def _load_files(root, split, split_seed, split_percent):
    fns = list(sorted([fn for fn in os.listdir(root) if fn.endswith(".pkl")]))
    fns = [os.path.join(root, fn) for fn in fns]
    split_idx = int(len(fns) * split_percent)

    if split == "dev":
        return fns[:2]
    if split == "all":
        return fns

    random.Random(split_seed).shuffle(fns)
    if split == "train":
        fns = fns[:split_idx]
    elif split == "test":
        fns = fns[split_idx:]
    else:
        assert False
    return fns

def _process_episodes(fns: list[str],
                      radius: float,
                      aug_interpolate: float,
                      aug_clusters: float,
                      env_cfg):
    episodes = []
    datas = []
    max_num_points = 0

    for fn in fns:
        data = load_episode(fn)

        # TODO(?): truncate if reward is available
        episode = []
        curr_waypoint = None
        curr_waypoint_step = 0
        waypoint_len = 0

        target_mode = data[0]["mode"]
        for t, step in enumerate(list(data)):
            mode = step["mode"]
            print(mode, step["action"])

            if mode == ActMode.ArmWaypoint:
                if data[t + 1]["mode"] == ActMode.ArmWaypoint:
                    print(f"Warninig: skip step {t} in {fn} because the next one is also waypoint")
                    continue
                assert data[t + 1]["mode"] == ActMode.Interpolate

                curr_waypoint_step = t
                waypoint_len = 0
                for k in range(t + 1, len(list(data))):
                    if data[k]["mode"] != ActMode.Interpolate:
                        target_mode = data[k]["mode"]
                        break
                    waypoint_len += 1
                assert waypoint_len > 0

                if env_cfg.is_sim:
                    action = data[t+waypoint_len]["action"]
                    action_pos = action[:3]
                    action_quat = action[3:7]
                    action_gripper = action[7]
                else:
                    action = data[t+waypoint_len]["obs"]
                    action_pos = action["arm_pos"]
                    action_quat = action["arm_quat"]
                    action_gripper = np.round(data[t+waypoint_len]["action"][7])

                # Handle quaternion symmetry:
                if action_quat[3] < 0:
                    np.negative(action_quat, out=action_quat)

                curr_waypoint = {
                    "pos": action_pos,
                    "euler": R.from_quat(action_quat).as_euler('xyz'),
                    "quat": action_quat,  # type: ignore
                    "gripper": action_gripper,
                    "click": step["click"],  # a single vector of length 3
                }

            elif mode == ActMode.BaseWaypoint:
                if data[t + 1]["mode"] == ActMode.BaseWaypoint:
                    print(f"Warning: skip step {t} in {fn} because the next one is also waypoint")
                    continue
                assert data[t + 1]["mode"] == ActMode.Interpolate

                curr_waypoint_step = t
                waypoint_len = 0
                for k in range(t + 1, len(list(data))):
                    if data[k]["mode"] != ActMode.Interpolate:
                        target_mode = data[k]["mode"]
                        break
                    waypoint_len += 1
                assert waypoint_len > 0

                if env_cfg.is_sim:
                    action = data[t+waypoint_len]["action"]
                    base_rot = action[10]
                    base_pos = np.array([action[8], action[9], 0.0])
                else:
                    action = data[t+waypoint_len]["obs"]["base_pose"]
                    base_rot = action[2]
                    base_pos = np.array([action[0], action[1], step["click"][2]])

                base_rot = R.from_euler("z", base_rot)

                base_euler = base_rot.as_euler("xyz")
                base_quat = base_rot.as_quat()
                if base_quat[3] < 0:
                    base_quat *= -1

                curr_waypoint = {
                    "pos": base_pos,
                    "euler": base_euler,
                    "quat": base_quat,  # type: ignore
                    "gripper": 0,
                    "click": step["click"],  # a single vector of length 3
                }

            if mode not in [ActMode.ArmWaypoint, ActMode.BaseWaypoint, ActMode.Interpolate]:
                # Skip dense timesteps and non-terminal timesteps
                continue

            #target_mode = data[curr_waypoint_step]["mode"]

            if mode == ActMode.Interpolate:
                assert waypoint_len > 0
                step["click"] = curr_waypoint["click"]
                progress = (t - curr_waypoint_step) / waypoint_len
                # Keep this timestep only if we are doing temporal augmentation
                if progress > aug_interpolate:
                    continue

            assert curr_waypoint is not None
            obs = step["obs"]
            points, colors = pcl_from_obs(obs, env_cfg)

            if "proprio" in step["obs"]:
                proprio = step["obs"]["proprio"]
            else:
                proprio = np.hstack((obs["arm_pos"], obs["arm_quat"], obs["gripper_pos"], obs["base_pose"]))

            # label clicks
            dist_to_click = np.linalg.norm(
                points - np.expand_dims(curr_waypoint["click"], axis=0), axis=1
            )

            click_idxs = dist_to_click <= radius
            user_clicks = np.zeros((len(points),)).astype(points.dtype)
            user_clicks[click_idxs] = 1.0

            if user_clicks.sum() < 300.0:
                continue

            assert user_clicks.sum() != 0

            processed_data = {
                # input
                "xyz": points,
                "xyz_color": colors,
                "proprio": proprio,
                # to predict
                "user_clicks": user_clicks,
                "dist_to_click": dist_to_click,
                "action_pos": curr_waypoint["pos"],
                "action_euler": curr_waypoint["euler"],
                "action_quat": curr_waypoint["quat"],
                "action_gripper": curr_waypoint["gripper"],
                "target_mode": target_mode.value,  # FIXME, this should be target mode
                "click": curr_waypoint["click"],  # FIXME, this should be target mode
            }
            #print(mode, 'click:', step["click"], 'curr_pos', step['obs']['arm_pos'], 'action_pos:', curr_waypoint["pos"], 'action_grip:', curr_waypoint["gripper"], target_mode)
            #print(fn, target_mode, step["click"], curr_waypoint["pos"])
            #print(fn, target_mode, step["click"], curr_waypoint["pos"])
            episode.append(processed_data)
            datas.append(processed_data)
            #max_num_points = max(max_num_points, points.shape[0])

            max_possible_clusters = 6 * 800  # upper bound: 6 clusters × 800 points
            augmented_length = points.shape[0] + (max_possible_clusters if aug_clusters else 0)
            max_num_points = max(max_num_points, augmented_length)

        episodes.append(episode)
        #print('--------')
    return datas, episodes, max_num_points


@dataclass
class PointCloudDatasetConfig:
    path: str = ""
    is_real: int = 0
    split_seed: int = 1
    split_percent: float = 0.85
    repeat: int = 1
    # data format
    radius: float = 0.05
    use_dist: int = 0
    use_color: int = 1
    # augmentation
    aug_interpolate: float = 0  # create data even when the current mode is interpolate
    aug_translate: int = 0
    aug_clusters: int = 0
    aug_rotate: float = 0

    def __post_init__(self):
        PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
        DATASETS = {
            "pillow_base_arm": os.path.join(PROJECT_ROOT, "data/dev_pillow_base_arm"),
            "pillow_wbc": os.path.join(PROJECT_ROOT, "data/dev_pillow_wbc"),
            "cube_base_arm": os.path.join(PROJECT_ROOT, "data/dev_cube_base_arm"),
            "cube_wbc": os.path.join(PROJECT_ROOT, "data/dev_cube_wbc"),
            "cabinet_base_arm": os.path.join(PROJECT_ROOT, "data/dev_cabinet_base_arm"),
            "cabinet_wbc": os.path.join(PROJECT_ROOT, "data/dev_cabinet_wbc"),
        }
        if self.path in DATASETS:
            self.path = DATASETS[self.path]


class PointCloudDataset(Dataset):
    def __init__(self, cfg: PointCloudDatasetConfig, use_euler: bool, split: str):
        assert split in ["train", "test", "dev", "all"]
        self.cfg = cfg
        self.use_euler = use_euler
        self.split = split
        self.env_cfg_path = os.path.join(cfg.path, "env_cfg.yaml")
        try:
            self.env_cfg = pyrallis.load(MujocoEnvConfig, open(self.env_cfg_path, "r"))
        except:
            from envs.common_real_env_cfg import RealEnvConfig
            self.env_cfg = pyrallis.load(RealEnvConfig, open(self.env_cfg_path, "r"))

        self.fns = _load_files(cfg.path, split, cfg.split_seed, cfg.split_percent)
        #print(f"Creating {split} dataset with {len(self.fns)} demos")

        self.datas, self.episodes, self.max_num_points = _process_episodes(
            self.fns,
            self.cfg.radius,
            self.cfg.aug_interpolate,
            self.cfg.aug_clusters,
            self.env_cfg
            )
        print(f"Total num of data item: {len(self.datas)}, max_num_points: {self.max_num_points}")

    def __len__(self):
        return len(self.datas) * self.cfg.repeat

    def __getitem__(self, index):
        """
        return:
            point_set: np.ndarray
            colors: np.ndarray
            user_clicked_labels: np.ndarray
            action_pos: np.ndarray
            action_quat: np.ndarray
            action_gripper: float
            proprio: np.ndarray
            mode: int
        """
        if self.cfg.repeat > 1:
            index = index % len(self.datas)

        data = self.datas[index]
        xyz = torch.from_numpy(data["xyz"]).float()
        colors = torch.from_numpy(data["xyz_color"]).float()
        user_clicked_labels = torch.from_numpy(data["user_clicks"]).long()

        if self.cfg.aug_clusters:
            click_point = torch.from_numpy(data["click"]).float()
            dist_to_click = torch.from_numpy(data["dist_to_click"]).float()
            if random.random() < 0.5:
                xyz, colors, user_clicked_labels, dist_to_click = augment_with_clusters(
                    xyz, colors, user_clicked_labels, dist_to_click, click_point
                )
        else:
            dist_to_click = torch.from_numpy(data["dist_to_click"]).float()


        # NOTE: we call fps outside in the training code
        # pad every data point to the same number of points
        num_padding_point = self.max_num_points - xyz.shape[0]
        padding_indices = np.random.choice(xyz.shape[0], num_padding_point, replace=True)
        indices = torch.from_numpy(np.concatenate((np.arange(xyz.shape[0]), padding_indices)))

        xyz = xyz[indices, :]
        colors = colors[indices, :]

        # these are prediction targets
        user_clicked_labels = user_clicked_labels[indices]
        assert user_clicked_labels.sum() != 0

        if self.cfg.use_dist:
            dist = self.cfg.radius - dist_to_click
            dist = dist[indices]
            user_clicked_labels = user_clicked_labels * dist
            assert user_clicked_labels.min() == 0
            assert user_clicked_labels.max() <= self.cfg.radius
            user_clicked_labels /= user_clicked_labels.max()

        action_pos = torch.from_numpy(data["action_pos"]).float()
        if self.use_euler:
            action_rot = torch.from_numpy(data["action_euler"]).float()
        else:
            action_rot = torch.from_numpy(data["action_quat"]).float()
        action_gripper = torch.tensor(data["action_gripper"], dtype=torch.float32)
        proprio = torch.from_numpy(data["proprio"]).float()
        target_mode = torch.tensor(data["target_mode"]).long()


        if self.cfg.aug_translate:
            xyz, colors, action_pos, proprio = augment_with_translation(
                xyz, colors, action_pos, proprio
            )
        if self.cfg.aug_rotate:
            xyz, action_pos, action_rot, proprio = augment_with_rotation(
                xyz, action_pos, action_rot, proprio, self.cfg.aug_rotate
            )

        if self.cfg.use_color:
            pcd = torch.cat((xyz, colors), 1)
        else:
            pcd = xyz

        return (
            pcd,
            proprio,
            user_clicked_labels,
            action_pos,
            action_rot,
            action_gripper,
            target_mode,
        )

    def save_vis(self, save_dir, render_gripper, fps: int, use_triton: bool):
        import open3d as o3d
        save_dir = os.path.join(self.cfg.path, save_dir)
        print(f"saving vis to {save_dir}, {self.cfg.aug_interpolate=}, {self.cfg.aug_translate=}")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        for i in range(len(self.datas)):
            pcd, proprio, clicks, action_pos, action_rot, _, _ = self[i]
            points, colors = pcd.split([3, 3], dim=1)

            if fps > 0:
                if use_triton:
                    # assert not self.cfg.use_dist, "not supported yet for viz"
                    # triton fps requires # points to be pow of 2
                    num_pad = common_utils.next_power_of_two(points.size(0))
                    pad_indices = np.random.choice(
                        points.shape[0], num_pad - points.size(0), replace=True
                    )
                    points = torch.cat([points, points[pad_indices]], dim=0)
                    colors = torch.cat([colors, colors[pad_indices]], dim=0)
                    clicks = torch.cat([clicks, clicks[pad_indices]], dim=0)

                    indices = triton_farthest_point_sample(
                        points.unsqueeze(0).cuda(), fps
                    ).cpu().squeeze()
                    print("# unique point:", len(set(indices.numpy().tolist())))
                else:
                    indices = farthest_point_sample(points.cuda().unsqueeze(0), fps).squeeze(0).cpu()

                points = points[indices]
                colors = colors[indices]
                clicks = clicks[indices]

            clicks = clicks.unsqueeze(1)
            red = torch.tensor([1, 0, 0], dtype=torch.float32)
            colors = red * clicks + colors * (1 - clicks)

            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points.numpy())
            point_cloud.colors = o3d.utility.Vector3dVector(colors.numpy())

            if render_gripper:
                # render target gripper
                gripper_vis = o3d.io.read_triangle_mesh(
                    "interactive_scripts/interactive_utils/franka.obj"
                )
                gripper_vis.paint_uniform_color([0.8, 0.0, 0.0])
                rotation_matrix = R.from_euler("xyz", action_rot).as_matrix()
                default_rot = R.from_euler("x", -np.pi / 2).as_matrix()
                rotation_matrix = rotation_matrix @ default_rot
                transform = np.eye(4)
                transform[:3, :3] = rotation_matrix
                transform[:, 3][:-1] = action_pos.numpy()
                gripper_vis.transform(transform)
                point_cloud += gripper_vis.sample_points_uniformly(number_of_points=1000)

                # render curr gripper
                gripper_vis = o3d.io.read_triangle_mesh(
                    "interactive_scripts/interactive_utils/franka.obj"
                )
                gripper_vis.paint_uniform_color([0.0, 0.0, 0.8])
                rotation_matrix = R.from_euler("xyz", proprio[3:6]).as_matrix()
                rotation_matrix = rotation_matrix @ default_rot
                transform = np.eye(4)
                transform[:3, :3] = rotation_matrix
                transform[:, 3][:-1] = proprio[:3].numpy()
                gripper_vis.transform(transform)
                point_cloud += gripper_vis.sample_points_uniformly(number_of_points=1000)

            path = os.path.join(save_dir, f"%05d.pcd" % i)
            print(f"saving to {path}")
            o3d.io.write_point_cloud(path, point_cloud)


def main():
    cfg = PointCloudDatasetConfig(
        #path="data/dev_wiping_wbc",
        path="data/dev_cube_wbc",
        #path="data/dev_remote_wbc",
        #path="data/dev_pillow_base_arm",
        #path="data/dev_cube_base_arm",
        #path="data/dev_cabinet_base_arm",
        is_real=0,
        radius=0.1,
        aug_interpolate=0,
        aug_translate=0,
        aug_clusters=1,
        aug_rotate=0,
        use_dist=1,
        # fps=1,
        # use_triton=1
    )
    dataset = PointCloudDataset(cfg, use_euler=False, split="all")
    d = dataset[0]
    print("target_mode", d[-1].item())
    user_clicked_labels = d[2] / d[2].sum()
    indices = user_clicked_labels > 0
    print("#positive:", user_clicked_labels[indices].size())
    print("click labels:", user_clicked_labels[indices])
    print("click labels max:", user_clicked_labels[indices].max())
    #dataset.save_vis("vis_dev", render_gripper=False, fps=0, use_triton=True)
    #dataset.save_vis("vis_dev", render_gripper=False, fps=1024, use_triton=True)

if __name__ == "__main__":
    main()
