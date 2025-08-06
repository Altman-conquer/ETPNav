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
import json
import os
import shutil
import sys
from typing import Any, Dict, List, Optional, Type

import attr
import cv2
import git
import magnum
import magnum as mn
import numpy as np
import quaternion

from matplotlib import pyplot as plt
from PIL import Image

import habitat
import habitat_sim
from habitat.config import Config
from habitat.core.registry import registry
from habitat_sim.utils import viz_utils as vut


data_path = "/home/zhandijia/DockerData/zhandijia-root/ETPNav/data"
print(f"data_path = {data_path}")
# @markdown Optionally configure the save path for video output:
output_directory = "examples/"
output_path = os.path.join('/home/zhandijia/DockerData/zhandijia-root/ETPNav', output_directory)


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
                cross_hair[0] - 2 : cross_hair[0] + 2,
                cross_hair[1] - 2 : cross_hair[1] + 2,
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
        },
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
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.1)
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
    "seed": 1,
    "enable_physics": True,
    "physics_config_file": "data/default.physics_config.json",
    "silent": False,
    "compute_shortest_path": False,
    "compute_action_shortest_path": False,
    "save_png": True,
}

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

def test(sim, rgbs=None, depths=None, cameras=None):
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

    W = H = 256
    hfov = 90
    K = np.array([
        [1 / np.tan(hfov / 2.), 0., 0., 0.],
        [0., 1 / np.tan(hfov / 2.), 0., 0.],
        [0., 0., 1, 0],
        [0., 0., 0, 1]])

    filter_using_height = False
    min_z = 0.8  # 地板以上
    max_z = 2.5  # 天花板以下

    all_points = []
    all_colors = []

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
        zs = world_coordinates[2, :]  # (N,)

        # 过滤地板和天花板
        if filter_using_height:
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
    z_values = merged_points[:, 2]  # 所有点的高度

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

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

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

    for _ in range(steps):
        # 获取当前传感器的观测
        observations = sim.get_sensor_observations()
        rgb_images.append(observations["rgb"])
        depth_images.append(observations["depth"])
        cameras.append(sim.agents[0].state)

        # 让代理旋转
        sim.agents[0].act("turn_right")

    return rgb_images, depth_images, cameras

def load_from_npy_folder(rgb_dir, depth_dir, pose_dir):
    import numpy as np
    import os
    import pickle

    rgbs, depths, cameras = [], [], []
    rgb_files = sorted(os.listdir(rgb_dir))
    for file in rgb_files:
        timestamp = file.replace('.pkl', '')
        with open(os.path.join(rgb_dir, f"{timestamp}.pkl"), "rb") as f:
            rgb = pickle.load(f)
        with open(os.path.join(depth_dir, f"{timestamp}.pkl"), "rb") as f:
            depth = pickle.load(f)
        with open(os.path.join(pose_dir, f"{timestamp}.pkl"), "rb") as f:
            pose = pickle.load(f)
        rgbs.append(rgb)
        depths.append(depth)
        cameras.append(pose)
    return rgbs, depths, cameras

# pcd = test(sim, rgbs, depths, cameras)
# down_project_pcd_to_2d(pcd, 'yz')

with habitat_sim.Simulator(cfg) as sim:
    # get_2d_point(sim, 'rgb', mn.Vector3(0, 0, 0.5))
    init_agent(sim)
    init_objects(sim)

    rgbs, depths, cameras = rotate_and_capture(sim, steps=3)
    # rgbs, depths, cameras = load_from_npy_folder("tmp/rgb_images", "tmp/depth_images", "tmp/poses")

    pcd = test(sim, rgbs, depths, cameras)
    down_project_pcd_to_2d(pcd, 'yz')

    # Visualize the scene after the chair is added into the scene.
    if make_video:
        simulate_and_make_vid(
            sim, None, "object-init", dt=1.0, open_vid=False
        )