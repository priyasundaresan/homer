import open3d as o3d
import os
import pyrallis
import numpy as np
from envs.utils.camera_utils import pcl_from_obs
from interactive_scripts.dataset_recorder import load_episode

# Globals to store state
current_idx = 0
demo = []
env_cfg = None
geometry_list = []
vis = None

def update_geometry():
    global current_idx, demo, env_cfg, vis, geometry_list

    # Save the current view control (camera)
    view_ctl = vis.get_view_control()
    camera_params = view_ctl.convert_to_pinhole_camera_parameters()

    # Remove geometries
    for g in geometry_list:
        try:
            vis.remove_geometry(g, reset_bounding_box=False)
        except Exception as e:
            print(f"Warning: could not remove geometry: {e}")
    geometry_list.clear()

    # Get current step
    step = demo[current_idx]
    obs = step["obs"]
    arm_pos = obs["arm_pos"]
    merged_points, merged_colors = pcl_from_obs(obs, env_cfg)

    if merged_points is None or merged_colors is None or len(merged_points) == 0:
        print(f"Step {current_idx}: Invalid or empty point cloud data.")
        return

    # Make point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack(merged_points))
    pcd.colors = o3d.utility.Vector3dVector(np.vstack(merged_colors))

    # Sphere at arm_pos
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    sphere.paint_uniform_color([1.0, 0.0, 0.0])
    sphere.translate(arm_pos)

    vis.add_geometry(pcd)
    vis.add_geometry(sphere)
    geometry_list.extend([pcd, sphere])

    # Restore the camera view
    vis.get_view_control().convert_from_pinhole_camera_parameters(camera_params)

    vis.update_renderer()
    print(f"Step {current_idx+1}/{len(demo)}: Showing {len(merged_points)} points.")

def next_frame(vis):
    global current_idx
    if current_idx < len(demo) - 1:
        current_idx += 30
        update_geometry()
    return False


def prev_frame(vis):
    global current_idx
    if current_idx > 0:
        current_idx -= 1
        update_geometry()
    return False

def visualize_pcl(episode_fn, _env_cfg):
    global demo, env_cfg, vis
    env_cfg = _env_cfg

    demo = load_episode(episode_fn)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    update_geometry()

    vis.register_key_callback(ord("N"), next_frame)
    vis.register_key_callback(ord("P"), prev_frame)

    print("Press 'N' for next frame, 'P' for previous.")
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    demo_dir = 'dev1'
    try:
        from envs.common_mj_env import MujocoEnvConfig
        env_cfg = pyrallis.load(MujocoEnvConfig, open(os.path.join(demo_dir, "env_cfg.yaml"), "r"))
    except:
        from envs.common_real_env_cfg import RealEnvConfig
        env_cfg = pyrallis.load(RealEnvConfig, open(os.path.join(demo_dir, "env_cfg.yaml"), "r"))

    visualize_pcl(f'{demo_dir}/demo00000.pkl', env_cfg)

