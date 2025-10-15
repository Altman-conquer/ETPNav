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
import os
import pickle

import cv2
import habitat_sim
import magnum as mn
import numpy as np
import numpy.linalg as LA
from PIL import Image
from habitat_sim.utils import viz_utils as vut
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R

from vlnce_baselines.map_navigation.map_utils import init_ins2cat_dict
from vlnce_baselines.map_navigation.rgb_map import rgb_map_habitat_tools
from vlnce_baselines.map_navigation.semantic_map import semantic_map_habitat_tools


def test_load_semantic_map():
    semantic_map = semantic_map_habitat_tools(saved_folder='tmp/semantic_map/', MIN_DEPTH=0.0,
                                              MAX_DEPTH=1.0)
    semantic_map.load_complete_map('/home/zhandijia/DockerData/zhandijia-root/ETPNav/data/logs/maps/BEV_semantic_map.npy')
    semantic_map.vis_info = {
            'nodes': np.array([[ 0.102541,    0.17162801, -0.185072  ], [1.0807754,  0.17162801, 0.02243072]]),
            'ghosts': np.array([[-0.61171241,  0.17162801, -0.23356644], [1.79416466, 0.17162801, 0.25389504], [ 2.16903947,  0.17162801, -0.22253835]]),
            'agent_angle': np.pi  * 0.25,
        }
    semantic_map.save_final_map(display_object_classes=['hall', 'table', 'open doorway', 'bathroom doorway', 'sink', 'doorway'])
    # semantic_map.find_path_to_area_around_object([0.1025409996509552, 0.17162801325321198, -0.18507200479507446],'sink', search_radius=25)

def test_load_rgb_map():
    rgb_map = rgb_map_habitat_tools(saved_folder='tmp/rgb_map/', MIN_DEPTH=0.0,
                                   MAX_DEPTH=10.0)
    rgb_map.load_complete_rgb_map('/home/zhandijia/DockerData/zhandijia-root/ETPNav/data/logs/maps/BEV_rgb_map.npy')
    rgb_map.vis_info = {
            'nodes': np.array([[ 0.102541, 0.17162801, -0.185072  ], [1.0807754,  0.17162801, 0.02243072]]),
            'ghosts': np.array([[-0.61171241,  0.17162801, -0.23356644], [1.79416466, 0.17162801, 0.25389504], [ 2.16903947,  0.17162801, -0.22253835]]),
            'agent_angle': np.pi  * 0.25,
        }
    rgb_map.save_final_map(ENLARGE_SIZE=2, display_object_classes=['hall', 'table', 'open doorway', 'bathroom doorway', 'sink', 'doorway'])
    # rgb_map.find_path_to_area_around_object([0.1025409996509552, 0.17162801325321198, -0.18507200479507446],'sink', search_radius=25)

# test_load_semantic_map()
test_load_rgb_map()

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
    # init_ins2cat_dict(sim)
    # scene_semantics = sim.semantic_scene  # set([i.name() for i in scene_semantics.categories])
    # ins2cat_dict = {
    #     int(obj.id.split("_")[-1]): obj.category.index() for obj in scene_semantics.objects}
    # get_2d_point(sim, 'rgb', mn.Vector3(0, 0, 0.5))
    init_agent(sim)  # scene_semantics.objects[0].category.index()
    # init_objects(sim)

    # vlmap = PointCloudVlmap()
    # rgbs, depths, depth_cameras = vlmap.rotate_and_capture(sim, steps=36)
    # vlmap.create_camera_map(rgbs, depths, depth_cameras)

    create_folder('tmp/rgb_map')
    create_folder('tmp/semantic_map')

    rgb_map = rgb_map_habitat_tools(saved_folder='tmp/rgb_map/', MIN_DEPTH=MIN_DEPTH, MAX_DEPTH=MAX_DEPTH)
    # rgbs, depths, semantics, depth_cameras = rgb_map.rotate_and_capture(sim, rotate_step=36, move_step=1)
    rgbs, depths, semantics, depth_cameras, _ = rgb_map.load_from_npy_folder("tmp/rgb_images", "tmp/depth_images",
                                                                             "tmp/semantic_images",
                                                                             "tmp/depth_poses", "tmp/rgb_poses", max_cnt=-1)

    # semantic_map = semantic_map_habitat_tools(saved_folder='tmp/semantic_map/', MIN_DEPTH=MIN_DEPTH, MAX_DEPTH=MAX_DEPTH)

    # rgb_map.get_detect_result(['/home/zhandijia/DockerData/zhandijia-root/ETPNav/tmp/rgb_images_png/20250904_103923.png'])
    detect_results = rgb_map.get_detect_result(rgbs)
    # semantics = semantic_map.detect(rgbs)

    from collections import Counter

    # 统计所有图片中所有检测到的类别
    all_classes = []
    for result in detect_results:
        for box in result.get('boxes', []):
            all_classes.append(box['cls'])
    class_counter = Counter(all_classes)
    print("类别统计：", class_counter)

    count_ = 0

    from tqdm import tqdm

    for rgb, depth, semantic, pose, detect_result in tqdm(zip(rgbs, depths, semantics, depth_cameras, detect_results),
                                                          total=len(rgbs), desc="Processing"):
        rgb_map.build_rgb_map(rgb, depth, detect_result['boxes'], pose, count_)
        # semantic_map.build_semantic_map(detect_result['boxes'], depth, semantic, pose, count_)
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
