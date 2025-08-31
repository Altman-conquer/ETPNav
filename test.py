# from vlnce_baselines.models.encoders.resnet_encoders import (
#     TorchVisionResNet50,
#     VlnResnetDepthEncoder,
#     CLIPEncoder,
# )
#
# def main():
#     depth_encoder = VlnResnetDepthEncoder(
#         observation_space,
#         output_size=model_config.DEPTH_ENCODER.output_size,
#         checkpoint=model_config.DEPTH_ENCODER.ddppo_checkpoint,
#         backbone=model_config.DEPTH_ENCODER.backbone,
#         spatial_output=model_config.spatial_output,
#     )
#
#     rgb_encoder = CLIPEncoder(self.device)
#
# if __name__ == '__main__':
#     main()


# @title Path Setup and Imports { display-mode: "form" }
# @markdown (double click to show code).

## [setup]
import gzip
import io
import json
import os
import shutil
import sys
from typing import Any, Dict, List, Optional, Type, Union
import open3d as o3d
import numpy as np
import os
import pickle

import requests
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
from mpl_toolkits.mplot3d import Axes3D

import attr
import cv2
import git
import magnum
import magnum as mn
import numpy as np
import quaternion
from scipy.spatial.transform import Rotation as R

from matplotlib import pyplot as plt
from PIL import Image

import habitat
import habitat_sim
from habitat.config import Config
from habitat.core.registry import registry
from habitat_sim.utils import viz_utils as vut
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from math import cos, sin, pi, floor, tan
import numpy.linalg as LA

data_path = "/home/zhandijia/DockerData/zhandijia-root/ETPNav/data"
print(f"data_path = {data_path}")
# @markdown Optionally configure the save path for video output:
output_directory = "examples/"
output_path = os.path.join('/home/zhandijia/DockerData/zhandijia-root/ETPNav', output_directory)

ins2cat_dict = None

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.add_argument(
        "--no-make-video", dest="make_video", action="store_false"
    )
    parser.set_defaults(show_video=True, make_video=True)
    args, _ = parser.parse_known_args()
    show_video = args.display
    display = args.display
    make_video = args.make_video
else:
    show_video = False
    make_video = False
    display = False

if make_video and not os.path.exists(output_path):
    os.makedirs(output_path)


def make_video_cv2(
        observations, cross_hair=None, prefix="", open_vid=True, fps=60
):
    sensor_keys = list(observations[0])
    videodims = observations[0][sensor_keys[0]].shape
    videodims = (videodims[1], videodims[0])  # flip to w,h order
    print(videodims)
    video_file = output_path + prefix + ".mp4"
    print("Encoding the video: %s " % video_file)
    writer = vut.get_fast_video_writer(video_file, fps=fps)
    for ob in observations:
        # If in RGB/RGBA format, remove the alpha channel
        rgb_im_1st_person = cv2.cvtColor(ob["rgb"], cv2.COLOR_RGBA2RGB)
        if cross_hair is not None:
            rgb_im_1st_person[
            cross_hair[0] - 2: cross_hair[0] + 2,
            cross_hair[1] - 2: cross_hair[1] + 2,
            ] = [255, 0, 0]

        if rgb_im_1st_person.shape[:2] != videodims:
            rgb_im_1st_person = cv2.resize(
                rgb_im_1st_person, videodims, interpolation=cv2.INTER_AREA
            )
        # write the 1st person observation to video
        writer.append_data(rgb_im_1st_person)
    writer.close()

    if open_vid:
        print("Displaying video")
        vut.display_video(video_file)


def simulate(sim, dt=1.0, get_frames=True):
    # simulate dt seconds at 60Hz to the nearest fixed timestep
    print("Simulating " + str(dt) + " world seconds.")
    observations = []
    start_time = sim.get_world_time()
    while sim.get_world_time() < start_time + dt:
        sim.step_physics(1.0 / 60.0)
        if get_frames:
            observations.append(sim.get_sensor_observations())
    return observations


# convenience wrapper for simulate and make_video_cv2
def simulate_and_make_vid(sim, crosshair, prefix, dt=1.0, open_vid=True):
    observations = simulate(sim, dt)
    make_video_cv2(observations, crosshair, prefix=prefix, open_vid=open_vid)


def display_sample(
        rgb_obs,
        semantic_obs=np.array([]),
        depth_obs=np.array([]),
        key_points=None,  # noqa: B006
):
    from habitat_sim.utils.common import d3_40_colors_rgb

    rgb_img = Image.fromarray(rgb_obs, mode="RGB")

    arr = [rgb_img]
    titles = ["rgb"]
    if semantic_obs.size != 0:
        semantic_img = Image.new(
            "P", (semantic_obs.shape[1], semantic_obs.shape[0])
        )
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        arr.append(semantic_img)
        titles.append("semantic")

    if depth_obs.size != 0:
        depth_img = Image.fromarray(
            (depth_obs / 10 * 255).astype(np.uint8), mode="L"
        )
        arr.append(depth_img)
        titles.append("depth")

    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        # plot points on images
        if key_points is not None:
            for point in key_points:
                plt.plot(
                    point[0], point[1], marker="o", markersize=10, alpha=0.8
                )
        plt.imshow(data)

    plt.show(block=False)


def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.default_agent_id = settings["default_agent_id"]
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.enable_physics = settings["enable_physics"]
    sim_cfg.physics_config_file = settings["physics_config_file"]

    # Note: all sensors must have the same resolution
    sensors = {
        "rgb": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
        },
        "depth": {
            "sensor_type": habitat_sim.SensorType.DEPTH,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
            "min_depth": 0.0,
            "max_depth": 10.0
        },
        "semantic": {
            "sensor_type": habitat_sim.SensorType.SEMANTIC,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
        }
    }

    sensor_specs = []
    for sensor_uuid, sensor_params in sensors.items():
        if settings[sensor_uuid]:
            sensor_spec = habitat_sim.SensorSpec()
            sensor_spec.uuid = sensor_uuid
            sensor_spec.sensor_type = sensor_params["sensor_type"]
            sensor_spec.resolution = sensor_params["resolution"]
            sensor_spec.position = sensor_params["position"]

            sensor_specs.append(sensor_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


settings = {
    "max_frames": 10,
    "width": 256,
    "height": 256,
    # "scene": "data/scene_datasets/coda/coda.glb",
    "scene": "data/scene_datasets/mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb",
    "default_agent_id": 0,
    "sensor_height": 1.5,  # Height of sensors in meters
    "rgb": True,  # RGB sensor
    "depth": True,  # Depth sensor
    "semantic": True,
    "seed": 1,
    "enable_physics": True,
    "physics_config_file": "data/default.physics_config.json",
    "silent": False,
    "compute_shortest_path": False,
    "compute_action_shortest_path": False,
    "save_png": True,
}

MIN_DEPTH = 0.0
MAX_DEPTH = 10.0

cfg = make_cfg(settings)


def init_agent(sim):
    # agent_pos = np.array([-0.15776923, 0.18244143, 0.2988735])
    agent_pos = np.array([0.1025409996509552, 0.17162801325321198, -0.18507200479507446])

    # Place the agent
    sim.agents[0].scene_node.translation = agent_pos
    agent_orientation_y = -40
    sim.agents[0].scene_node.rotation = mn.Quaternion.rotation(
        mn.Deg(agent_orientation_y), mn.Vector3(0, 1.0, 0)
    )


cfg.sim_cfg.default_agent_id = 0
with habitat_sim.Simulator(cfg) as sim:
    init_agent(sim)
    if make_video:
        # Visualize the agent's initial position
        simulate_and_make_vid(
            sim, None, "sim-init", dt=1.0, open_vid=show_video
        )


def remove_all_objects(sim):
    for obj_id in sim.get_existing_object_ids():
        sim.remove_object(obj_id)


def set_object_in_front_of_agent(sim, obj_id, z_offset=-1.5):
    r"""
    Adds an object in front of the agent at some distance.
    """
    agent_transform = sim.agents[0].scene_node.transformation_matrix()
    obj_translation = agent_transform.transform_point(
        np.array([0, 0, z_offset])
    )
    sim.set_translation(obj_translation, obj_id)

    # obj_node = sim.get_object_scene_node(obj_id)
    # xform_bb = habitat_sim.geo.get_transformed_bb(
    #     obj_node.cumulative_bb, obj_node.transformation
    # )
    #
    # # also account for collision margin of the scene
    # scene_collision_margin = 0.04
    # y_translation = mn.Vector3(
    #     0, xform_bb.size_y() / 2.0 + scene_collision_margin, 0
    # )
    # sim.set_translation(y_translation + sim.get_translation(obj_id), obj_id)

    # scale the object
    # scale_matrix = mn.Matrix4.scaling(mn.Vector3(2.0))
    # obj_node.transformation = obj_node.transformation @ scale_matrix


def init_objects(sim):
    # Manager of Object Attributes Templates
    obj_attr_mgr = sim.get_object_template_manager()
    obj_attr_mgr.load_configs(
        str(os.path.join(data_path, "test_assets/objects"))
    )

    # Add a chair into the scene.
    obj_path = "test_assets/objects/sphere"
    chair_template_id = obj_attr_mgr.load_object_configs(
        str(os.path.join(data_path, obj_path))
    )[0]
    chair_attr = obj_attr_mgr.get_template_by_ID(chair_template_id)
    # chair_attr.render_asset_handle = None
    obj_attr_mgr.register_template(chair_attr)

    # Object's initial position 3m away from the agent.
    object_id = sim.add_object_by_handle(chair_attr.handle)
    set_object_in_front_of_agent(sim, object_id, -3.0)
    sim.set_object_motion_type(
        habitat_sim.physics.MotionType.STATIC, object_id
    )

    # Object's final position 7m away from the agent
    # goal_id = sim.add_object_by_handle(chair_attr.handle)
    # set_object_in_front_of_agent(sim, goal_id, -7.0)
    # sim.set_object_motion_type(habitat_sim.physics.MotionType.STATIC, goal_id)

    # return object_id, goal_id


def get_2d_point(sim, sensor_name, point_3d):
    # get the scene render camera and sensor object
    visual_sensor = sim._sensors[sensor_name]
    scene_graph = sim.get_active_scene_graph()
    scene_graph.set_default_render_camera_parameters(visual_sensor._sensor_object)
    render_camera = scene_graph.get_default_render_camera()

    # use the camera and projection matrices to transform the point onto the near plane
    projected_point_3d = render_camera.projection_matrix.transform_point(
        render_camera.camera_matrix.transform_point(point_3d)
    )
    # convert the 3D near plane point to integer pixel space
    point_2d = mn.Vector2(projected_point_3d[0], -projected_point_3d[1])
    point_2d = point_2d / render_camera.projection_size()[0]
    point_2d += mn.Vector2(0.5)
    point_2d *= render_camera.viewport
    return mn.Vector2i(point_2d)


def cvt_pose_vec2tf(pos_quat_vec: np.ndarray) -> np.ndarray:
    """
    pos_quat_vec: (px, py, pz, qx, qy, qz, qw)
    """
    pose_tf = np.eye(4)
    pose_tf[:3, 3] = pos_quat_vec[:3].flatten()
    rot = R.from_quat(pos_quat_vec[3:].flatten())
    pose_tf[:3, :3] = rot.as_matrix()
    return pose_tf


def test(sim, rgbs=None, depths=None, cameras=None, save_img: bool = False, hfov: float = 90):
    import os
    import shutil
    import open3d as o3d
    import numpy as np
    import cv2
    import quaternion

    if rgbs is None or depths is None or cameras is None:
        depths = []
        rgbs = []
        cameras = [sim.agents[0].state]
        depth = sim.get_sensor_observations()['depth']
        rgb = sim.get_sensor_observations()['rgb']
        depths.append(depth)
        rgbs.append(rgb)

    hfov = np.deg2rad(hfov)

    W = H = 256
    # K = np.array([
    #     [1 / np.tan(hfov / 2.), 0., 0., 0.],
    #     [0., 1 / np.tan(hfov / 2.), 0., 0.],
    #     [0., 0., 1, 0],
    #     [0., 0., 0, 1]])
    K = np.array([
        [W / (2.0 * np.tan(np.deg2rad(hfov / 2))), 0., 0., 0.],
        [0., W / (2.0 * np.tan(np.deg2rad(hfov / 2))), 0., 0.],
        [0., 0., 1, 0],
        [0., 0., 0, 1]])

    filter_using_height = False

    all_points = []
    all_colors = []

    # camera_pose_tfs = [cvt_pose_vec2tf(x) for x in self.camera_pose_tfs]

    for i in range(len(rgbs)):
        depth = depths[i].reshape(1, W, W)
        rgb = rgbs[i][:, :, :3]  # 取RGB通道
        if rgb.shape[:2] != depth.shape[1:]:
            rgb = cv2.resize(rgb, (depth.shape[2], depth.shape[1]), interpolation=cv2.INTER_AREA)

        xs, ys = np.meshgrid(np.linspace(-1, 1, W), np.linspace(1, -1, W))
        xs = xs.reshape(1, W, W)
        ys = ys.reshape(1, W, W)

        xys = np.vstack((xs * depth, ys * depth, -depth, np.ones(depth.shape)))
        xys = xys.reshape(4, -1)
        xy_c0 = np.matmul(np.linalg.inv(K), xys)

        quaternion_0 = cameras[i].rotation
        translation_0 = cameras[i].position
        rotation_0 = quaternion.as_rotation_matrix(quaternion_0)
        T_world_camera0 = np.eye(4)
        T_world_camera0[0:3, 0:3] = rotation_0
        T_world_camera0[0:3, 3] = translation_0

        world_coordinates = np.matmul(T_world_camera0, xy_c0)  # shape: (4, N)
        points = world_coordinates[:3, :].T  # (N, 3)
        zs = world_coordinates[1, :]  # (N,)

        # 过滤地板和天花板
        if filter_using_height:
            agent_height = cameras[i].position[1]
            min_z = agent_height - 0.8
            max_z = agent_height + 0.5
            mask = (zs > min_z) & (zs < max_z)
        else:
            mask = np.ones_like(zs, dtype=bool)
        filtered_points = points[mask]

        # 计算颜色
        xs_img = ((xs.flatten() + 1) * (W - 1) / 2).astype(np.int32)
        ys_img = ((1 - ys.flatten()) * (W - 1) / 2).astype(np.int32)
        xs_img = xs_img[mask]
        ys_img = ys_img[mask]
        color = rgb[ys_img, xs_img] / 255.0  # 归一化到[0,1]
        all_points.append(filtered_points)
        all_colors.append(color)

    merged_points = np.concatenate(all_points, axis=0)
    merged_colors = np.concatenate(all_colors, axis=0)

    # 假设 merged_points 是 (N, 3) 的点云数组
    z_values = merged_points[:, 1]  # 所有点的高度

    # 设置高度区间，比如每 0.1 米一个区间
    bins = np.arange(z_values.min(), z_values.max() + 0.1, 0.1)
    hist, bin_edges = np.histogram(z_values, bins=bins)

    # 输出每个高度区间的点数
    for i in range(len(hist)):
        print(f"高度区间 {bin_edges[i]:.2f} ~ {bin_edges[i + 1]:.2f} 米: 点数 {hist[i]}")

    # 用open3d保存带颜色点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged_points)
    pcd.colors = o3d.utility.Vector3dVector(merged_colors)
    pcd = pcd.voxel_down_sample(voxel_size=0.05)

    os.makedirs("tmp", exist_ok=True)
    output_file = "tmp/point_cloud_merged.ply"
    o3d.io.write_point_cloud(output_file, pcd)

    # 假设 points 是 (N, 3) 的点云数组
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

    # 保存深度和RGB图像
    if save_img:
        if os.path.exists("tmp/depth_images"):
            shutil.rmtree("tmp/depth_images")
        if os.path.exists("tmp/rgb_images"):
            shutil.rmtree("tmp/rgb_images")
        os.makedirs("tmp/depth_images", exist_ok=True)
        os.makedirs("tmp/rgb_images", exist_ok=True)
        for idx, depth in enumerate(depths):
            depth_image = (depth * 255 / np.max(depth)).astype(np.uint8)
            cv2.imwrite(f"tmp/depth_images/depth_{idx:03d}.png", depth_image)
        for idx, rgb in enumerate(rgbs):
            rgb_image = cv2.cvtColor(rgb, cv2.COLOR_RGBA2RGB)
            cv2.imwrite(f"tmp/rgb_images/rgb_{idx:03d}.png", rgb_image)

    return pcd


# def create_global_point_cloud(rgb_list, depth_list, pose_list, voxel_size=0.02):
#     hfov = 90
#     intrinsics = np.array([
#         [1 / np.tan(hfov / 2.), 0., 0., 0.],
#         [0., 1 / np.tan(hfov / 2.), 0., 0.],
#         [0., 0., 1, 0],
#         [0., 0., 0, 1]])
#
#     height, width = depth_list[0].shape[0], depth_list[0].shape[1]
#     intrinsics = o3d.open3d.camera.PinholeCameraIntrinsic(
#         width,
#         height,
#         fx=intrinsics[0][0],
#         fy=intrinsics[1][1],
#         cx=intrinsics[0][2],
#         cy=intrinsics[1][2],
#     )
#
#     for depth_img, camera in zip(depth_list, pose_list):
#         depth_img = depth_img.reshape(depth_img.shape[0], depth_img.shape[0])
#
#         quaternion_0 = camera.rotation
#         translation_0 = camera.position
#         rotation_0 = quaternion.as_rotation_matrix(quaternion_0)
#         T_world_camera0 = np.eye(4)
#         T_world_camera0[0:3, 0:3] = rotation_0
#         T_world_camera0[0:3, 3] = translation_0
#
#         # 只用深度图生成点云
#         pcd = o3d.geometry.PointCloud.create_from_depth_image(
#             o3d.geometry.Image(depth_img), intrinsics
#         )
#         pcd.transform(T_world_camera0)
#         global_pcd += pcd
#
#     # 降采样
#     global_pcd = global_pcd.voxel_down_sample(voxel_size=voxel_size)
#     return global_pcd

def down_project_pcd_to_2d(pcd, plane='xy', grid_size=0.05):
    import numpy as np
    import cv2
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    if plane == 'xy':
        x, y = points[:, 0], points[:, 1]
    elif plane == 'xz':
        x, y = points[:, 0], points[:, 2]
    elif plane == 'yz':
        x, y = points[:, 1], points[:, 2]
    else:
        raise ValueError("plane must be 'xy', 'xz', or 'yz'")

    # 归一化到正区间
    x_min, y_min = x.min(), y.min()
    x = x - x_min
    y = y - y_min

    # 网格尺寸
    x_max, y_max = x.max(), y.max()
    img_w = int(np.ceil(x_max / grid_size)) + 1
    img_h = int(np.ceil(y_max / grid_size)) + 1

    img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    count = np.zeros((img_h, img_w), dtype=np.int32)
    color_sum = np.zeros((img_h, img_w, 3), dtype=np.float32)

    # 投影点云到2D网格，累加颜色
    for i in range(points.shape[0]):
        xi = int(x[i] / grid_size)
        yi = int(y[i] / grid_size)
        color_sum[yi, xi] += colors[i]
        count[yi, xi] += 1

    # 计算平均颜色
    mask = count > 0
    img[mask] = (color_sum[mask] / count[mask, None] * 255).astype(np.uint8)

    cv2.imwrite('tmp/down_projected_2d.png', img)
    return img


def rotate_and_capture(sim, steps=36):
    """
    让代理旋转360度，并捕获所有看到的RGB和深度图像。

    参数:
    - sim: Habitat模拟器实例。
    - steps: 将360度分成的步数（每步旋转角度为360/steps）。

    返回:
    - rgb_images: 包含所有RGB图像的列表。
    - depth_images: 包含所有深度图像的列表。
    """
    rgb_images = []
    depth_images = []
    cameras = []

    # 每步旋转的角度
    angle_per_step = 360 / steps

    for _ in range(10):
        for _ in range(steps):
            # 获取当前传感器的观测
            observations = sim.get_sensor_observations()
            rgb_images.append(observations["rgb"])
            depth_images.append(observations["depth"])
            cameras.append(sim.agents[0].state.sensor_states['depth'])

            # sensor_state = sim.agents[0].state.sensor_states['depth']
            # cameras.append([
            #     sensor_state.position[0],
            #     sensor_state.position[1],
            #     sensor_state.position[2],
            #     sensor_state.rotation.x,
            #     sensor_state.rotation.y,
            #     sensor_state.rotation.z,
            #     sensor_state.rotation.w,
            # ])
            # 让代理旋转
            sim.agents[0].act("turn_right")

        sim.agents[0].act("move_forward")

    return rgb_images, depth_images, cameras


def load_from_npy_folder(rgb_dir, depth_dir, depth_pose_dir, rgb_pose_dir):
    rgbs, depths, rgb_cameras, depth_cameras = [], [], [], []
    rgb_files = sorted(os.listdir(rgb_dir))
    for file in rgb_files:
        timestamp = file.replace('.pkl', '')
        with open(os.path.join(rgb_dir, f"{timestamp}.pkl"), "rb") as f:
            rgb = pickle.load(f)
        with open(os.path.join(depth_dir, f"{timestamp}.pkl"), "rb") as f:
            depth = pickle.load(f)
        with open(os.path.join(depth_pose_dir, f"{timestamp}.pkl"), "rb") as f:
            depth_pose = pickle.load(f)
        with open(os.path.join(rgb_pose_dir, f"{timestamp}.pkl"), "rb") as f:
            rgb_pose = pickle.load(f)

        assert not np.array_equal(depth_pose, rgb_pose)

        rgbs.append(rgb)
        depths.append(depth)
        depth_cameras.append(depth_pose)
        rgb_cameras.append(rgb_pose)
    return rgbs, depths, depth_cameras, rgb_cameras


# pcd = test(sim, rgbs, depths, cameras)
# down_project_pcd_to_2d(pcd, 'yz')


def get_agent_pose(agent_state):
    agent_pos = agent_state.position
    agent_rot = agent_state.rotation
    heading_vector = quaternion_rotate_vector(
        agent_rot.inverse(), np.array([0, 0, -1]))
    phi = cartesian_to_polar(
        -heading_vector[2], heading_vector[0])[1]
    angle = phi
    print(f'agent position = {agent_pos}, angle = {angle}')
    pose = (agent_pos[0], agent_pos[2], angle)
    return pose


def project_semantic_pixels_to_world_coords(sseg_img,
                                            current_depth,
                                            current_pose,
                                            gap=2,
                                            FOV=79,
                                            cx=320,
                                            cy=240,
                                            theta_x=0.0,
                                            resolution_x=640,
                                            resolution_y=480,
                                            ignored_classes=[],
                                            sensor_height=1.5):
    """
    Project pixels in sseg_img into world frame given depth image current_depth and camera pose current_pose.
    (u, v) = KRT(XYZ)
    """
    from core import cfg

    # camera intrinsic matrix
    radian = FOV * pi / 180.
    focal_length = cx / tan(radian / 2)
    K = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]])
    inv_K = LA.inv(K)
    # first compute the rotation and translation from current frame to goal frame
    # then compute the transformation matrix from goal frame to current frame
    # thransformation matrix is the camera2's extrinsic matrix
    tx, tz, theta = current_pose

    R_y = np.array([[cos(theta), 0, sin(theta)], [0, 1, 0],
                    [-sin(theta), 0, cos(theta)]])
    # used when I tilt the camera up/down
    R_x = np.array([[1, 0, 0], [0, cos(theta_x), -sin(theta_x)],
                    [0, sin(theta_x), cos(theta_x)]])
    R = R_y.dot(R_x)
    T = np.array([tx, 0, tz])
    transformation_matrix = np.empty((3, 4))
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = T

    # build the point matrix
    x = range(0, resolution_x, gap)
    y = range(0, resolution_y, gap)
    xv, yv = np.meshgrid(np.array(x), np.array(y))
    Z = current_depth[yv.flatten(),
    xv.flatten()].reshape(yv.shape[0], yv.shape[1])
    points_4d = np.ones((yv.shape[0], yv.shape[1], 4), np.float32)
    points_4d[:, :, 0] = xv
    points_4d[:, :, 1] = yv
    points_4d[:, :, 2] = Z
    points_4d = np.transpose(points_4d, (2, 0, 1)).reshape((4, -1))  # 4 x N

    # apply intrinsic matrix
    points_4d[[0, 1, 3], :] = inv_K.dot(points_4d[[0, 1, 3], :])
    points_4d[0, :] = points_4d[0, :] * points_4d[2, :]
    points_4d[1, :] = points_4d[1, :] * points_4d[2, :]

    # transform kp1_4d from camera1(current) frame to camera2(goal) frame through transformation matrix
    points_3d = transformation_matrix.dot(points_4d)

    # reverse y-dim and add sensor height
    points_3d[1, :] = points_3d[1, :] * -1 + sensor_height

    # ignore some artifacts points with depth == 0
    depth_points = current_depth[yv.flatten(), xv.flatten()].flatten()
    good = np.logical_and(depth_points > cfg.SENSOR.DEPTH_MIN,
                          depth_points < cfg.SENSOR.DEPTH_MAX)

    points_3d = points_3d[:, good]

    # pick x-row and z-row
    sseg_points = sseg_img[yv.flatten(), xv.flatten()].flatten()
    sseg_points = sseg_points[good]

    # ignore some classes points
    for c in ignored_classes:
        good = (sseg_points != c)
        sseg_points = sseg_points[good]
        points_3d = points_3d[:, good]

    return points_3d, sseg_points.astype(int)

def project_pixels_to_world_coords(pixel_points: np.array, rgb_image, current_depth, current_pose, gap=2, FOV=79,
                                  cx=320,
                                  cy=240,
                                  theta_x=0.0,
                                  resolution_x=640,
                                  resolution_y=480,
                                  ignored_colors=[],
                                  sensor_height=1.5):
    from math import cos, sin, pi, tan
    import numpy.linalg as LA

    radian = FOV * pi / 180.
    focal_length = cx / tan(radian / 2)
    K = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]])
    inv_K = LA.inv(K)
    tx, tz, theta = current_pose

    R_y = np.array([[cos(theta), 0, sin(theta)], [0, 1, 0],
                    [-sin(theta), 0, cos(theta)]])
    R_x = np.array([[1, 0, 0], [0, cos(theta_x), -sin(theta_x)],
                    [0, sin(theta_x), cos(theta_x)]])
    R = R_y.dot(R_x)
    T = np.array([tx, 0, tz])
    transformation_matrix = np.empty((3, 4))
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = T

    # pixel_points: (N, 2), 每行是(u, v)
    u = pixel_points[:, 0]
    v = pixel_points[:, 1]
    Z = current_depth[v, u]
    points_4d = np.ones((4, len(u)), np.float32)
    points_4d[0, :] = u
    points_4d[1, :] = v
    points_4d[2, :] = Z

    points_4d[[0, 1, 3], :] = inv_K.dot(points_4d[[0, 1, 3], :])
    points_4d[0, :] = points_4d[0, :] * points_4d[2, :]
    points_4d[1, :] = points_4d[1, :] * points_4d[2, :]

    points_3d = transformation_matrix.dot(points_4d)
    points_3d[1, :] = points_3d[1, :] * -1 + sensor_height

    depth_points = Z
    good = np.logical_and(depth_points > MIN_DEPTH,
                          depth_points < MAX_DEPTH)

    rgb_points = rgb_image[v, u].reshape(-1, 3)

    # 可选：忽略某些颜色
    # for color in ignored_colors:
    #     mask = ~np.all(rgb_points == color, axis=1)
    #     rgb_points = rgb_points[mask]
    #     points_3d = points_3d[:, mask]

    return points_3d, rgb_points, good

def project_rgb_pixels_to_world_coords(rgb_image,
                                       current_depth,
                                       current_pose,
                                       gap=2,
                                       FOV=79,
                                       cx=320,
                                       cy=240,
                                       theta_x=0.0,
                                       resolution_x=640,
                                       resolution_y=480,
                                       ignored_colors=[],
                                       sensor_height=1.5):
    """
    将rgb_image中的像素投影到世界坐标系，返回3D点和对应的RGB颜色。
    """
    radian = FOV * pi / 180.
    focal_length = cx / tan(radian / 2)
    K = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]])
    inv_K = LA.inv(K)
    tx, tz, theta = current_pose

    R_y = np.array([[cos(theta), 0, sin(theta)], [0, 1, 0],
                    [-sin(theta), 0, cos(theta)]])
    R_x = np.array([[1, 0, 0], [0, cos(theta_x), -sin(theta_x)],
                    [0, sin(theta_x), cos(theta_x)]])
    R = R_y.dot(R_x)
    T = np.array([tx, 0, tz])
    transformation_matrix = np.empty((3, 4))
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = T

    x = range(0, resolution_x, gap)
    y = range(0, resolution_y, gap)
    xv, yv = np.meshgrid(np.array(x), np.array(y))
    Z = current_depth[yv.flatten(), xv.flatten()].reshape(yv.shape[0], yv.shape[1])
    points_4d = np.ones((yv.shape[0], yv.shape[1], 4), np.float32)
    points_4d[:, :, 0] = xv
    points_4d[:, :, 1] = yv
    points_4d[:, :, 2] = Z
    points_4d = np.transpose(points_4d, (2, 0, 1)).reshape((4, -1))  # 4 x N

    points_4d[[0, 1, 3], :] = inv_K.dot(points_4d[[0, 1, 3], :])
    points_4d[0, :] = points_4d[0, :] * points_4d[2, :]
    points_4d[1, :] = points_4d[1, :] * points_4d[2, :]

    points_3d = transformation_matrix.dot(points_4d)
    points_3d[1, :] = points_3d[1, :] * -1 + sensor_height

    depth_points = current_depth[yv.flatten(), xv.flatten()].flatten()
    good = np.logical_and(depth_points > MIN_DEPTH,
                          depth_points < MAX_DEPTH)
    points_3d = points_3d[:, good]

    rgb_points = rgb_image[yv.flatten(), xv.flatten()].reshape(-1, 3)
    rgb_points = rgb_points[good]

    # 可选：忽略某些颜色
    # for color in ignored_colors:
    #     mask = ~np.all(rgb_points == color, axis=1)
    #     rgb_points = rgb_points[mask]
    #     points_3d = points_3d[:, mask]

    return points_3d, rgb_points, good


def find_first_nonzero_elem_per_row(mat):
    H, W = mat.shape
    x = np.linspace(0, W - 1, W)
    y = np.linspace(0, H - 1, H)
    xv, yv = np.meshgrid(x, y)

    xv[mat == 0] = 0
    min_idx_nonzero_per_row = np.max(xv, axis=1).astype(int)

    yv = yv[:, 0].astype(int)

    result = mat[yv, min_idx_nonzero_per_row]
    return result


d3_41_colors_rgb: np.ndarray = np.array(
    [
        [0, 0, 0],
        [31, 119, 180],
        [174, 199, 232],
        [255, 127, 14],
        [255, 187, 120],
        [44, 160, 44],
        [152, 223, 138],
        [214, 39, 40],
        [255, 152, 150],
        [148, 103, 189],
        [197, 176, 213],
        [140, 86, 75],
        [196, 156, 148],
        [227, 119, 194],
        [247, 182, 210],
        [127, 127, 127],
        [199, 199, 199],
        [188, 189, 34],
        [219, 219, 141],
        [23, 190, 207],
        [158, 218, 229],
        [57, 59, 121],
        [82, 84, 163],
        [107, 110, 207],
        [156, 158, 222],
        [99, 121, 57],
        [140, 162, 82],
        [181, 207, 107],
        [206, 219, 156],
        [140, 109, 49],
        [189, 158, 57],
        [231, 186, 82],
        [231, 203, 148],
        [132, 60, 57],
        [173, 73, 74],
        [214, 97, 107],
        [231, 150, 156],
        [123, 65, 115],
        [165, 81, 148],
        [206, 109, 189],
        [222, 158, 214],
        [255, 255, 255]
    ],
    dtype=np.uint8,
)


def colormap(rgb=False):
    color_list = np.array(
        [
            0.000,
            0.447,
            0.741,
            0.850,
            0.325,
            0.098,
            0.929,
            0.694,
            0.125,
            0.494,
            0.184,
            0.556,
            0.466,
            0.674,
            0.188,
            0.301,
            0.745,
            0.933,
            0.635,
            0.078,
            0.184,
            0.300,
            0.300,
            0.300,
            0.600,
            0.600,
            0.600,
            1.000,
            0.000,
            0.000,
            1.000,
            0.500,
            0.000,
            0.749,
            0.749,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.333,
            0.333,
            0.000,
            0.333,
            0.667,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            0.333,
            0.000,
            0.667,
            0.667,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            1.000,
            0.000,
            0.000,
            0.333,
            0.500,
            0.000,
            0.667,
            0.500,
            0.000,
            1.000,
            0.500,
            0.333,
            0.000,
            0.500,
            0.333,
            0.333,
            0.500,
            0.333,
            0.667,
            0.500,
            0.333,
            1.000,
            0.500,
            0.667,
            0.000,
            0.500,
            0.667,
            0.333,
            0.500,
            0.667,
            0.667,
            0.500,
            0.667,
            1.000,
            0.500,
            1.000,
            0.000,
            0.500,
            1.000,
            0.333,
            0.500,
            1.000,
            0.667,
            0.500,
            1.000,
            1.000,
            0.500,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.333,
            0.333,
            1.000,
            0.333,
            0.667,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.667,
            0.333,
            1.000,
            0.667,
            0.667,
            1.000,
            0.667,
            1.000,
            1.000,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            1.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.143,
            0.143,
            0.143,
            0.286,
            0.286,
            0.286,
            0.429,
            0.429,
            0.429,
            0.571,
            0.571,
            0.571,
            0.714,
            0.714,
            0.714,
            0.857,
            0.857,
            0.857,
            1.000,
            1.000,
            1.000,
        ]
    ).astype(np.float32)
    color_list = (color_list.reshape((-1, 3)) * 255).astype(np.uint8)
    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list


# if # of classes is <= 41, flag_small_categories is True
def apply_color_to_map(semantic_map, dataset='MP3D'):
    """ convert semantic map semantic_map into a colorful visualization color_semantic_map"""
    assert len(semantic_map.shape) == 2
    if dataset == 'MP3D':
        COLOR = d3_41_colors_rgb
        num_classes = 41
    elif dataset == 'HM3D':
        COLOR = colormap(rgb=True)
        num_classes = 300
    else:
        raise NotImplementedError(
            f"Dataset {dataset} not currently supported.")

    H, W = semantic_map.shape
    color_semantic_map = np.zeros((H, W, 3), dtype='uint8')
    for i in range(num_classes):
        if dataset == 'MP3D':
            color_semantic_map[semantic_map == i] = COLOR[i]
        elif dataset == 'HM3D':
            color_semantic_map[semantic_map == i] = COLOR[i % len(COLOR), 0:3]
    return color_semantic_map


class semantic_map_habitat_tools:
    """ class used to build semantic maps of the scenes.
    It takes dense observations of the environment and project pixels to the ground.
    """

    def __init__(self, saved_folder):
        from core import cfg

        self.scene_name = ''
        self.cell_size = cfg.SEM_MAP.CELL_SIZE
        self.step_size = 1000
        self.map_boundary = 5
        self.detector = None
        self.saved_folder = saved_folder

        self.IGNORED_CLASS = [0, 1]  # ceiling class is ignored

        # ==================================== initialize 4d grid =================================
        self.min_X = -cfg.SEM_MAP.WORLD_SIZE
        self.max_X = cfg.SEM_MAP.WORLD_SIZE
        self.min_Z = -cfg.SEM_MAP.WORLD_SIZE
        self.max_Z = cfg.SEM_MAP.WORLD_SIZE
        self.min_Y = 0.0
        self.max_Y = cfg.SENSOR.AGENT_HEIGHT + self.cell_size

        self.x_grid = np.arange(self.min_X, self.max_X, self.cell_size)
        self.z_grid = np.arange(self.min_Z, self.max_Z, self.cell_size)
        self.y_grid = np.arange(self.min_Y, self.max_Y, self.cell_size)

        self.THRESHOLD_HIGH = len(self.y_grid)

        self.four_dim_grid = np.zeros(
            (len(self.z_grid), len(self.y_grid) + 1,
             len(self.x_grid), cfg.SEM_MAP.GRID_CLASS_SIZE),
            dtype=np.int16)  # x, y, z, C

        # ===================================
        self.H, self.W = len(self.z_grid), len(self.x_grid)
        self.min_x_coord = self.W - 1
        self.max_x_coord = 0
        self.min_z_coord = self.H - 1
        self.max_z_coord = 0
        self.max_y_coord = 0

        self.pcd = o3d.geometry.PointCloud()  # 初始化为 None，后续可赋值为 open3d.geometry.PointCloud()

    def convert_insseg_to_sseg(self, insseg, ins2cat_dict):
        """
        convert instance segmentation image InsSeg (generated by Habitat Simulator) into Semantic segmentation image SSeg,
        given the mapping from instance to category ins2cat_dict.
        """
        ins_id_list = list(ins2cat_dict.keys())
        sseg = np.zeros(insseg.shape, dtype=np.int16)
        for ins_id in ins_id_list:
            sseg = np.where(insseg == ins_id, ins2cat_dict[ins_id], sseg)
        return sseg

    def build_semantic_map(self, rgb_img, depth_img, insseg_img, pose, step_):
        """ update semantic map with observations rgb_img, depth_img, sseg_img and robot pose."""
        global ins2cat_dict

        sem_map_pose = (pose[0], -pose[1], -pose[2])  # x, z, theta

        sseg_img = self.convert_insseg_to_sseg(insseg_img, ins2cat_dict)

        xyz_points, sseg_points = project_semantic_pixels_to_world_coords(
            sseg_img, depth_img, sem_map_pose, gap=2, FOV=90, cx=128, cy=128, resolution_x=256, resolution_y=256,
            ignored_classes=self.IGNORED_CLASS)

        new_point_cloud = o3d.geometry.PointCloud()
        new_point_cloud.points = o3d.utility.Vector3dVector(xyz_points.T)
        self.pcd += new_point_cloud

        mask_X = np.logical_and(xyz_points[0, :] > self.min_X,
                                xyz_points[0, :] < self.max_X)
        mask_Z = np.logical_and(xyz_points[2, :] > self.min_Z,
                                xyz_points[2, :] < self.max_Z)
        mask_XYZ = np.logical_and.reduce((mask_X, mask_Z))
        xyz_points = xyz_points[:, mask_XYZ]
        sseg_points = sseg_points[mask_XYZ]

        x_coord = np.floor(
            (xyz_points[0, :] - self.min_X) / self.cell_size).astype(int)
        y_coord = np.digitize(xyz_points[1, :], self.y_grid)
        z_coord = (self.H - 1) - np.floor(
            (xyz_points[2, :] - self.min_Z) / self.cell_size).astype(int)

        if x_coord.shape[0] > 0:
            self.four_dim_grid[z_coord, y_coord, x_coord, sseg_points] += 1

            # update the weights for the local map
            self.min_x_coord = min(max(np.min(x_coord) - self.map_boundary, 0),
                                   self.min_x_coord)
            self.max_x_coord = max(
                min(np.max(x_coord) + self.map_boundary, self.W - 1),
                self.max_x_coord)
            self.min_z_coord = min(max(np.min(z_coord) - self.map_boundary, 0),
                                   self.min_z_coord)
            self.max_z_coord = max(
                min(np.max(z_coord) + self.map_boundary, self.H - 1),
                self.max_z_coord)

            self.max_y_coord = max(np.max(y_coord), self.max_y_coord)

        if step_ % self.step_size == 0:
            self.get_semantic_map(step_)

    def get_semantic_map(self, step_):
        """ get the built semantic map. """
        smaller_four_dim_grid = self.four_dim_grid[self.min_z_coord:self.max_z_coord + 1, 0:self.THRESHOLD_HIGH,
                                self.min_x_coord:self.max_x_coord + 1, :]
        # argmax over the category axis
        zyx_grid = np.argmax(smaller_four_dim_grid, axis=3)
        # swap y dim to the last axis
        zxy_grid = np.swapaxes(zyx_grid, 1, 2)
        L, M, N = zxy_grid.shape
        zxy_grid = zxy_grid.reshape(L * M, N)

        semantic_map = find_first_nonzero_elem_per_row(zxy_grid)
        semantic_map = semantic_map.reshape(L, M)
        color_semantic_map = apply_color_to_map(semantic_map)

        if semantic_map.shape[0] > 0:
            self.save_sem_map_through_plt(
                color_semantic_map,
                f'{self.saved_folder}/step_{step_}_semantic.jpg')

    def save_sem_map_through_plt(self, img, name):
        """ save the figure img at directory 'name' using matplotlib"""
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.tight_layout()
        fig.savefig(name)
        plt.close()

    def save_final_map(self, ENLARGE_SIZE=5):
        """ save the built semantic map to a figure."""
        smaller_four_dim_grid = self.four_dim_grid[self.min_z_coord:self.max_z_coord + 1, 0:self.THRESHOLD_HIGH,
                                self.min_x_coord:self.max_x_coord + 1, :]
        # argmax over the category axis
        zyx_grid = np.argmax(smaller_four_dim_grid, axis=3)
        # swap y dim to the last axis
        zxy_grid = np.swapaxes(zyx_grid, 1, 2)
        L, M, N = zxy_grid.shape
        zxy_grid = zxy_grid.reshape(L * M, N)

        semantic_map = find_first_nonzero_elem_per_row(zxy_grid)
        semantic_map = semantic_map.reshape(L, M)

        map_dict = {}
        map_dict['min_x'] = self.min_x_coord
        map_dict['max_x'] = self.max_x_coord
        map_dict['min_z'] = self.min_z_coord
        map_dict['max_z'] = self.max_z_coord
        map_dict['min_X'] = self.min_X
        map_dict['max_X'] = self.max_X
        map_dict['min_Z'] = self.min_Z
        map_dict['max_Z'] = self.max_Z
        map_dict['W'] = self.W
        map_dict['H'] = self.H
        map_dict['semantic_map'] = semantic_map
        print(f'semantic_map.shape = {semantic_map.shape}')
        np.save(f'{self.saved_folder}/BEV_semantic_map.npy', map_dict)

        semantic_map = cv2.resize(
            semantic_map,
            (int(semantic_map.shape[1] * ENLARGE_SIZE),
             int(semantic_map.shape[0] * ENLARGE_SIZE)),
            interpolation=cv2.INTER_NEAREST)
        color_semantic_map = apply_color_to_map(semantic_map)
        self.save_sem_map_through_plt(color_semantic_map,
                                 f'{self.saved_folder}/final_semantic_map.jpg')


class rgb_map_habitat_tools:
    """ 用于构建场景RGB地图的类，记录每个cell的平均颜色 """

    def __init__(self, saved_folder):
        from core import cfg

        self.scene_name = ''
        self.cell_size = cfg.SEM_MAP.CELL_SIZE
        self.step_size = 1000
        self.map_boundary = 5
        self.saved_folder = saved_folder

        self.min_X = -cfg.SEM_MAP.WORLD_SIZE
        self.max_X = cfg.SEM_MAP.WORLD_SIZE
        self.min_Z = -cfg.SEM_MAP.WORLD_SIZE
        self.max_Z = cfg.SEM_MAP.WORLD_SIZE
        self.min_Y = 0.0
        self.max_Y = cfg.SENSOR.AGENT_HEIGHT + self.cell_size

        self.x_grid = np.arange(self.min_X, self.max_X, self.cell_size)
        self.z_grid = np.arange(self.min_Z, self.max_Z, self.cell_size)
        self.y_grid = np.arange(self.min_Y, self.max_Y, self.cell_size)
        self.THRESHOLD_HIGH = len(self.y_grid)

        # 用于累加RGB和计数
        self.four_dim_grid_sum = np.zeros(
            (len(self.z_grid), len(self.y_grid) + 1, len(self.x_grid), 3), dtype=np.float32)
        self.four_dim_grid_count = np.zeros(
            (len(self.z_grid), len(self.y_grid) + 1, len(self.x_grid)), dtype=np.int32)

        self.H, self.W = len(self.z_grid), len(self.x_grid)
        self.min_x_coord = self.W - 1
        self.max_x_coord = 0
        self.min_z_coord = self.H - 1
        self.max_z_coord = 0
        self.max_y_coord = 0

        self.object_map = [] # {'position': (x, y, z), 'label': label, 'conf': conf}

        self.pcd = o3d.geometry.PointCloud()

    # object_map: [{'position': (x, y, z), 'label': label, ...}, ...]
    @staticmethod
    def deduplicate_objects(object_map, eps=0.2):
        unique_objects = []
        if not object_map:
            return unique_objects
        positions = np.array([obj['position'] for obj in object_map])
        labels = np.array([obj['label'] for obj in object_map])
        confs = np.array([obj.get('conf', 1.0) for obj in object_map])
        for inst_id in np.unique(labels):
            mask = (labels == inst_id)
            if np.sum(mask) == 0:
                continue
            db = DBSCAN(eps=eps, min_samples=1).fit(positions[mask])
            cluster_labels = db.labels_
            for label in set(cluster_labels):
                cluster_points = positions[mask][cluster_labels == label]

                if len(cluster_points) == 0:
                    raise ValueError("cluster_confs 为空，无法计算cluster_confs.max()。")

                center = cluster_points.mean(axis=0)
                cluster_confs = confs[mask][cluster_labels == label]
                mean_conf = cluster_confs.max()
                unique_objects.append({'position': center, 'label': inst_id, 'conf': mean_conf})
        return unique_objects

    def load_from_npy_folder(self, rgb_dir, depth_dir, semantic_dir, depth_pose_dir, rgb_pose_dir):
        rgbs, depths, semantics, rgb_cameras, depth_cameras = [], [], [], [], []
        rgb_files = sorted(os.listdir(rgb_dir))
        for file in rgb_files:
            timestamp = file.replace('.pkl', '')
            with open(os.path.join(rgb_dir, f"{timestamp}.pkl"), "rb") as f:
                rgb = pickle.load(f)
            with open(os.path.join(depth_dir, f"{timestamp}.pkl"), "rb") as f:
                depth = pickle.load(f)
                depth = depth * (MAX_DEPTH - MIN_DEPTH) + MIN_DEPTH
            with open(os.path.join(semantic_dir, f"{timestamp}.pkl"), "rb") as f:
                semantic = pickle.load(f)
            with open(os.path.join(depth_pose_dir, f"{timestamp}.pkl"), "rb") as f:
                depth_pose = pickle.load(f)
            with open(os.path.join(rgb_pose_dir, f"{timestamp}.pkl"), "rb") as f:
                rgb_pose = pickle.load(f)

            # 如果rgb和depth尺寸不一致，则缩放rgb到depth的尺寸
            if rgb.shape[:2] != depth.shape[:2]:
                rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA)

            assert depth_pose == rgb_pose

            rgbs.append(rgb)
            depths.append(depth)
            semantics.append(semantic)
            depth_cameras.append(depth_pose)
            rgb_cameras.append(rgb_pose)
        return rgbs, depths, semantics, depth_cameras, rgb_cameras

    def rotate_and_capture(self, sim, rotate_step=36, move_step=1):
        """
        让代理旋转360度，并捕获所有看到的RGB和深度图像。

        参数:
        - sim: Habitat模拟器实例。
        - steps: 将360度分成的步数（每步旋转角度为360/steps）。

        返回:
        - rgb_images: 包含所有RGB图像的列表。
        - depth_images: 包含所有深度图像的列表。
        """
        rgb_images = []
        depth_images = []
        semantic_images = []
        cameras = []

        for _ in range(move_step):
            for _ in range(rotate_step):
                observations = sim.get_sensor_observations()
                rgb_images.append(observations["rgb"][:, :, :3])
                depth_images.append(observations["depth"])
                semantic_images.append(observations["semantic"])
                cameras.append(get_agent_pose(sim.agents[0].state.sensor_states['depth']))

                # 让代理旋转
                sim.agents[0].act("turn_right")

            for _ in range(5):
                sim.agents[0].act("move_forward")

        if os.path.exists("tmp/depth_images"):
            shutil.rmtree("tmp/depth_images")
        if os.path.exists("tmp/rgb_images"):
            shutil.rmtree("tmp/rgb_images")
        os.makedirs("tmp/depth_images", exist_ok=True)
        os.makedirs("tmp/rgb_images", exist_ok=True)
        for idx, depth in enumerate(depth_images):
            depth_image = (depth * 255 / np.max(depth)).astype(np.uint8)
            cv2.imwrite(f"tmp/depth_images/depth_{idx:03d}.png", depth_image)
        for idx, rgb in enumerate(rgb_images):
            cv2.imwrite(f"tmp/rgb_images/rgb_{idx:03d}.png", rgb)

        return rgb_images, depth_images, semantic_images, cameras

    def get_detect_result(self, rgb_imgs: list):
        def prepare_image(img: Union[str, np.ndarray, Image.Image]) -> bytes:
            if isinstance(img, str):
                image = Image.open(img).convert("RGB")
            elif isinstance(img, np.ndarray):
                image = Image.fromarray(img.astype(np.uint8)).convert("RGB")
            elif isinstance(img, Image.Image):
                image = img.convert("RGB")
            else:
                raise ValueError("不支持的图片类型")
            buf = io.BytesIO()
            image.save(buf, format="JPEG")
            return buf.getvalue()

        def detect(images: List[Union[str, np.ndarray, Image.Image]], extra_class=None,
                   server_url="http://127.0.0.1:8000/detect/"):
            files = []
            for idx, img in enumerate(images):
                img_bytes = prepare_image(img)
                files.append(("files", (f"image{idx}.jpg", img_bytes, "image/jpeg")))
            data = []
            if extra_class:
                data = [("extra_class", cls) for cls in extra_class]
            response = requests.post(server_url, files=files, data=data)
            response.raise_for_status()
            return response.json()

        return detect(rgb_imgs)

    def build_rgb_map_slow(self, rgb_img, depth_img, detect_results: List[dict], pose, step_):
        """ 用观测rgb_img和depth_img更新RGB地图 """
        gap = 2
        resolution_x = 256

        map_pose = (pose[0], -pose[1], -pose[2])
        xyz_points, rgb_points, goods = project_rgb_pixels_to_world_coords(
            rgb_img, depth_img, map_pose, gap=gap, FOV=90, cx=128, cy=128, resolution_x=resolution_x, resolution_y=256)

        # for detect_result in detect_results:
        #     center = [int((detect_result['xyxy'][0] + detect_result['xyxy'][2]) / 2), int((detect_result['xyxy'][1] + detect_result['xyxy'][3]) / 2)]
        #     center = np.array([center])
        #
        #     center_points, center_rgbs, center_goods = project_pixels_to_world_coords(center, rgb_img, depth_img, map_pose, gap=gap, FOV=90, cx=128, cy=128, resolution_x=resolution_x, resolution_y=256)
        #
        #     self.object_map.append({
        #         'position': center_points[:3, 0],
        #         'label': detect_result['cls'],
        #         'conf': detect_result['conf']
        #     })
        #
        # self.object_map = self.deduplicate_objects(self.object_map)

        # new_point_cloud = o3d.geometry.PointCloud()
        # new_point_cloud.points = o3d.utility.Vector3dVector(xyz_points.T)
        # self.pcd += new_point_cloud

        mask_X = np.logical_and(xyz_points[0, :] > self.min_X, xyz_points[0, :] < self.max_X)
        mask_Z = np.logical_and(xyz_points[2, :] > self.min_Z, xyz_points[2, :] < self.max_Z)
        mask_XZ = np.logical_and(mask_X, mask_Z)
        xyz_points = xyz_points[:, mask_XZ]
        rgb_points = rgb_points[mask_XZ]

        x_coord = np.floor((xyz_points[0, :] - self.min_X) / self.cell_size).astype(int)
        y_coord = np.digitize(xyz_points[1, :], self.y_grid)
        z_coord = (self.H - 1) - np.floor((xyz_points[2, :] - self.min_Z) / self.cell_size).astype(int)

        for i in range(x_coord.shape[0]):
            self.four_dim_grid_sum[z_coord[i], y_coord[i], x_coord[i]] += rgb_points[i]
            self.four_dim_grid_count[z_coord[i], y_coord[i], x_coord[i]] += 1

        # 更新局部地图边界
        if x_coord.shape[0] > 0:
            self.min_x_coord = min(max(np.min(x_coord) - self.map_boundary, 0), self.min_x_coord)
            self.max_x_coord = max(min(np.max(x_coord) + self.map_boundary, self.W - 1), self.max_x_coord)
            self.min_z_coord = min(max(np.min(z_coord) - self.map_boundary, 0), self.min_z_coord)
            self.max_z_coord = max(min(np.max(z_coord) + self.map_boundary, self.H - 1), self.max_z_coord)
            self.max_y_coord = max(np.max(y_coord), self.max_y_coord)

        if step_ % self.step_size == 0:
            self.get_rgb_map(step_)

    # faster version
    def build_rgb_map(self, rgb_img, depth_img, detect_results: List[dict], pose, step_):
        """ 用观测rgb_img和depth_img更新RGB地图 """
        gap = 2
        resolution_x = 256

        map_pose = (pose[0], -pose[1], -pose[2])
        xyz_points, rgb_points, goods = project_rgb_pixels_to_world_coords(
            rgb_img, depth_img, map_pose, gap=gap, FOV=90, cx=128, cy=128,
            resolution_x=resolution_x, resolution_y=256)


        for detect_result in detect_results:
            center = [int((detect_result['xyxy'][0] + detect_result['xyxy'][2]) / 2), int((detect_result['xyxy'][1] + detect_result['xyxy'][3]) / 2)]
            center = np.array([center])

            center_points, center_rgbs, center_goods = project_pixels_to_world_coords(center, rgb_img, depth_img, map_pose, gap=gap, FOV=90, cx=128, cy=128, resolution_x=resolution_x, resolution_y=256)

            self.object_map.append({
                'position': center_points[:3, 0],
                'label': detect_result['cls'],
                'conf': detect_result['conf']
            })

        # 向量化边界过滤
        mask = ((xyz_points[0, :] >= self.min_X) & (xyz_points[0, :] < self.max_X) &
                (xyz_points[2, :] >= self.min_Z) & (xyz_points[2, :] < self.max_Z))

        if not np.any(mask):
            return

        xyz_points = xyz_points[:, mask]
        rgb_points = rgb_points[mask]

        # 向量化坐标计算
        x_coord = np.floor((xyz_points[0, :] - self.min_X) / self.cell_size).astype(np.int32)
        y_coord = np.digitize(xyz_points[1, :], self.y_grid)
        z_coord = (self.H - 1) - np.floor((xyz_points[2, :] - self.min_Z) / self.cell_size).astype(np.int32)

        # 边界检查，避免索引越界
        valid_mask = ((x_coord >= 0) & (x_coord < self.W) &
                      (z_coord >= 0) & (z_coord < self.H) &
                      (y_coord >= 0) & (y_coord < len(self.y_grid) + 1))

        x_coord = x_coord[valid_mask]
        y_coord = y_coord[valid_mask]
        z_coord = z_coord[valid_mask]
        rgb_points = rgb_points[valid_mask]

        if len(x_coord) == 0:
            return

        # 向量化累加操作 - 关键优化
        indices = (z_coord, y_coord, x_coord)
        np.add.at(self.four_dim_grid_sum, indices, rgb_points)
        np.add.at(self.four_dim_grid_count, indices, 1)

        # 向量化边界更新
        self.min_x_coord = min(max(np.min(x_coord) - self.map_boundary, 0), self.min_x_coord)
        self.max_x_coord = max(min(np.max(x_coord) + self.map_boundary, self.W - 1), self.max_x_coord)
        self.min_z_coord = min(max(np.min(z_coord) - self.map_boundary, 0), self.min_z_coord)
        self.max_z_coord = max(min(np.max(z_coord) + self.map_boundary, self.H - 1), self.max_z_coord)
        self.max_y_coord = max(np.max(y_coord), self.max_y_coord)

        if step_ % self.step_size == 0:
            self.get_rgb_map(step_)

    def get_rgb_map(self, step_):
        """ 获取当前构建的RGB地图 """
        self.object_map = self.deduplicate_objects(self.object_map)


        grid_sum = self.four_dim_grid_sum[self.min_z_coord:self.max_z_coord + 1, 0:self.THRESHOLD_HIGH,
                   self.min_x_coord:self.max_x_coord + 1, :]
        grid_count = self.four_dim_grid_count[self.min_z_coord:self.max_z_coord + 1, 0:self.THRESHOLD_HIGH,
                     self.min_x_coord:self.max_x_coord + 1]
        # 取y方向最大计数的cell
        zxy_grid = np.argmax(grid_count, axis=1)
        L, M = zxy_grid.shape
        rgb_map = np.zeros((L, M, 3), dtype=np.uint8)
        for i in range(L):
            for j in range(M):
                y_idx = zxy_grid[i, j]
                count = grid_count[i, y_idx, j]
                if count > 0:
                    rgb = grid_sum[i, y_idx, j] / count
                    rgb_map[i, j] = np.clip(rgb, 0, 255)
        plt.imsave(f'{self.saved_folder}/step_{step_}_rgb.jpg', rgb_map)

    # def save_final_map(self, ENLARGE_SIZE=5):
    #     """ 保存最终RGB地图 """
    #     grid_sum = self.four_dim_grid_sum[self.min_z_coord:self.max_z_coord + 1, 0:self.THRESHOLD_HIGH,
    #                self.min_x_coord:self.max_x_coord + 1, :]
    #     grid_count = self.four_dim_grid_count[self.min_z_coord:self.max_z_coord + 1, 0:self.THRESHOLD_HIGH,
    #                  self.min_x_coord:self.max_x_coord + 1]
    #     zxy_grid = np.argmax(grid_count, axis=1)
    #     L, M = zxy_grid.shape
    #     rgb_map = np.zeros((L, M, 3), dtype=np.uint8)
    #     for i in range(L):
    #         for j in range(M):
    #             y_idx = zxy_grid[i, j]
    #             count = grid_count[i, y_idx, j]
    #             if count > 0:
    #                 rgb = grid_sum[i, y_idx, j] / count
    #                 rgb_map[i, j] = np.clip(rgb, 0, 255)
    #     rgb_map = cv2.resize(rgb_map, (int(rgb_map.shape[1] * ENLARGE_SIZE), int(rgb_map.shape[0] * ENLARGE_SIZE)),
    #                          interpolation=cv2.INTER_NEAREST)
    #     plt.imsave(f'{self.saved_folder}/final_rgb_map.jpg', rgb_map)

    def save_final_map(self, ENLARGE_SIZE=5):
        """ 保存最终RGB地图并绘制检测到的物体 """
        grid_sum = self.four_dim_grid_sum[self.min_z_coord:self.max_z_coord + 1, 0:self.THRESHOLD_HIGH,
                   self.min_x_coord:self.max_x_coord + 1, :]
        grid_count = self.four_dim_grid_count[self.min_z_coord:self.max_z_coord + 1, 0:self.THRESHOLD_HIGH,
                     self.min_x_coord:self.max_x_coord + 1]
        zxy_grid = np.argmax(grid_count, axis=1)
        L, M = zxy_grid.shape
        rgb_map = np.zeros((L, M, 3), dtype=np.uint8)

        for i in range(L):
            for j in range(M):
                y_idx = zxy_grid[i, j]
                count = grid_count[i, y_idx, j]
                if count > 0:
                    rgb = grid_sum[i, y_idx, j] / count
                    rgb_map[i, j] = np.clip(rgb, 0, 255)

        # 放大地图
        rgb_map = cv2.resize(rgb_map, (int(rgb_map.shape[1] * ENLARGE_SIZE), int(rgb_map.shape[0] * ENLARGE_SIZE)),
                             interpolation=cv2.INTER_NEAREST)

        # 在地图上绘制物体
        for obj in self.object_map:
            pos = obj['position']
            label = obj['label']
            conf = obj.get('conf', 1.0)

            # 将世界坐标转换为地图坐标
            x_map = int((pos[0] - self.min_X) / self.cell_size - self.min_x_coord) * ENLARGE_SIZE
            z_map = int((self.H - 1 - (pos[2] - self.min_Z) / self.cell_size) - self.min_z_coord) * ENLARGE_SIZE

            # 确保坐标在地图范围内
            if 0 <= x_map < rgb_map.shape[1] and 0 <= z_map < rgb_map.shape[0]:
                # 绘制圆点标记物体位置
                cv2.circle(rgb_map, (x_map, z_map), 5 * ENLARGE_SIZE, (255, 0, 0), -1)

                # 添加文本标签
                text = f"{label}({conf:.2f})"
                cv2.putText(rgb_map, text, (x_map + 10, z_map - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5 * ENLARGE_SIZE, (255, 255, 255), 2)

        plt.imsave(f'{self.saved_folder}/final_rgb_map_with_objects.jpg', rgb_map)


def create_folder(folder_name, clean_up=False):
    """ create folder with directory folder_name.

    If the folder exists before creation, setup clean_up to True to remove files in the folder.
    """
    flag_exist = os.path.isdir(folder_name)
    if not flag_exist:
        print('{} folder does not exist, so create one.'.format(folder_name))
        os.makedirs(folder_name)
    else:
        print('{} folder already exists, so do nothing.'.format(folder_name))
        if clean_up:
            os.system('rm {}/*.png'.format(folder_name))
            os.system('rm {}/*.npy'.format(folder_name))
            os.system('rm {}/*.jpg'.format(folder_name))


with habitat_sim.Simulator(cfg) as sim:
    scene_semantics = sim.semantic_scene # set([i.name() for i in scene_semantics.categories])
    ins2cat_dict = {
        int(obj.id.split("_")[-1]): obj.category.index() for obj in scene_semantics.objects}
    # get_2d_point(sim, 'rgb', mn.Vector3(0, 0, 0.5))
    init_agent(sim) # scene_semantics.objects[0].category.index()
    # init_objects(sim)

    # vlmap = PointCloudVlmap()
    # rgbs, depths, depth_cameras = vlmap.rotate_and_capture(sim, steps=36)
    # vlmap.create_camera_map(rgbs, depths, depth_cameras)

    create_folder('tmp/rgb_map')
    create_folder('tmp/semantic_map')

    rgb_map = rgb_map_habitat_tools(saved_folder='tmp/rgb_map/')
    # rgbs, depths, semantics, depth_cameras = rgb_map.rotate_and_capture(sim, rotate_step=36, move_step=1)
    rgbs, depths, semantics, depth_cameras, _ = rgb_map.load_from_npy_folder("tmp/rgb_images", "tmp/depth_images", "tmp/semantic_images",
                                                                  "tmp/depth_poses", "tmp/rgb_poses")

    semantic_map = semantic_map_habitat_tools(saved_folder='tmp/semantic_map/')

    detect_results = rgb_map.get_detect_result(rgbs)

    count_ = 0

    from tqdm import tqdm

    for rgb, depth, semantic, pose, detect_result in tqdm(zip(rgbs, depths, semantics, depth_cameras, detect_results),
                                                          total=len(rgbs), desc="Processing"):
        rgb_map.build_rgb_map(rgb, depth, detect_result['boxes'], pose, count_)
        # semantic_map.build_semantic_map(rgb, depth, semantic, pose, count_)
        count_ += 1
    rgb_map.save_final_map()
    # semantic_map.save_final_map()

    # rgbs, depths, depth_cameras, _ = load_from_npy_folder("tmp/rgb_images", "tmp/depth_images", "tmp/depth_poses", "tmp/rgb_poses")
    # pcd = test(sim, rgbs, depths, depth_cameras, save_img=False)
    # pcd = create_global_point_cloud(rgbs, depths, cameras, voxel_size=0.05)

    # os.makedirs("tmp", exist_ok=True)
    # output_file = "tmp/point_cloud_merged.ply"
    # o3d.io.write_point_cloud(output_file, pcd)
    # down_project_pcd_to_2d(pcd, 'yz')

    # Visualize the scene after the chair is added into the scene.
    # if make_video:
    #     simulate_and_make_vid(
    #         sim, None, "object-init", dt=1.0, open_vid=False
    #     )
