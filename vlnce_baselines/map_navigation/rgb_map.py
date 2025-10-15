import io
import os
import pickle
import shutil
import traceback
from datetime import datetime
from typing import Union, List

import cv2
import numpy as np
import open3d as o3d
import requests
from PIL import Image
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
from math import pi, tan, cos, sin
from matplotlib import pyplot as plt
from numpy import linalg as LA

from vlnce_baselines.map_navigation.map_utils import deduplicate_objects


class rgb_map_habitat_tools:
    """ 用于构建场景RGB地图的类，记录每个cell的平均颜色 """

    def __init__(self, saved_folder, MIN_DEPTH, MAX_DEPTH):
        from core import cfg

        self.scene_name = ''
        self.cell_size = cfg.SEM_MAP.CELL_SIZE
        self.step_size = 1000
        self.map_boundary = 5
        self.saved_folder = saved_folder

        self.MIN_DEPTH = MIN_DEPTH
        self.MAX_DEPTH = MAX_DEPTH

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

        self.object_map = []  # {'position': (x, y, z), 'label': label, 'conf': conf}
        self.vis_info = None
        self.cropped_agent_info = None

        self.pcd = o3d.geometry.PointCloud()

    def load_from_npy_folder(self, rgb_dir, depth_dir, semantic_dir, depth_pose_dir, rgb_pose_dir, max_cnt=-1):
        rgbs, depths, semantics, rgb_cameras, depth_cameras = [], [], [], [], []
        rgb_files = sorted(os.listdir(rgb_dir))
        for file in rgb_files[:len(rgb_files) if max_cnt == -1 else max_cnt]:
            # for file in rgb_files[42:43]:
            timestamp = file.replace('.pkl', '')
            with open(os.path.join(rgb_dir, f"{timestamp}.pkl"), "rb") as f:
                rgb = pickle.load(f)
            with open(os.path.join(depth_dir, f"{timestamp}.pkl"), "rb") as f:
                depth = pickle.load(f)
                depth = depth * (self.MAX_DEPTH - self.MIN_DEPTH) + self.MIN_DEPTH
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

    def get_agent_pose(self, agent_state):
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

    def project_rgb_pixels_to_world_coords(self, rgb_image,
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
        good = np.logical_and(depth_points > self.MIN_DEPTH,
                              depth_points < self.MAX_DEPTH)
        points_3d = points_3d[:, good]

        rgb_points = rgb_image[yv.flatten(), xv.flatten()].reshape(-1, 3)
        rgb_points = rgb_points[good]

        # 可选：忽略某些颜色
        # for color in ignored_colors:
        #     mask = ~np.all(rgb_points == color, axis=1)
        #     rgb_points = rgb_points[mask]
        #     points_3d = points_3d[:, mask]

        return points_3d, rgb_points, good

    def project_pixels_to_world_coords(self, pixel_points: np.array, rgb_image, current_depth, current_pose, gap=2,
                                       FOV=79,
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
        good = np.logical_and(depth_points > self.MIN_DEPTH,
                              depth_points < self.MAX_DEPTH)

        rgb_points = rgb_image[v, u].reshape(-1, 3)

        # 可选：忽略某些颜色
        # for color in ignored_colors:
        #     mask = ~np.all(rgb_points == color, axis=1)
        #     rgb_points = rgb_points[mask]
        #     points_3d = points_3d[:, mask]

        return points_3d, rgb_points, good

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
                cameras.append(self.get_agent_pose(sim.agents[0].state.sensor_states['depth']))

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

    def get_detect_result(self, rgb_imgs: list, extra_class: list = None):
        # def prepare_image(img: Union[str, np.ndarray, Image.Image]) -> bytes:
        #     if isinstance(img, str):
        #         image = Image.open(img).convert("RGB")
        #     elif isinstance(img, np.ndarray):
        #         image = Image.fromarray(img.astype(np.uint8)).convert("RGB")
        #     elif isinstance(img, Image.Image):
        #         image = img.convert("RGB")
        #     else:
        #         raise ValueError("不支持的图片类型")
        #     buf = io.BytesIO()
        #     image.save(buf, format="JPEG")
        #     return buf.getvalue()
        #
        # def detect(images: List[Union[str, np.ndarray, Image.Image]], extra_class,
        #            server_url="http://127.0.0.1:8000/detect/"):
        #     files = []
        #     for idx, img in enumerate(images):
        #         img_bytes = prepare_image(img)
        #         files.append(("files", (f"image{idx}.jpg", img_bytes, "image/jpeg")))
        #     # data = []
        #     # if extra_class:
        #     #     data = [("extra_class", cls) for cls in extra_class]
        #     # response = requests.post(server_url, files=files, data=data)
        #     response = requests.post(server_url, files=files)
        #     response.raise_for_status()
        #     return response.json()
        #
        # return detect(rgb_imgs, extra_class=extra_class)
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

        files = []
        for idx, img in enumerate(rgb_imgs):
            img_bytes = prepare_image(img)
            files.append(("files", (f"image{idx}.jpg", img_bytes, "image/jpeg")))
        data = []
        if extra_class:
            data = [("extra_class", cls) for cls in extra_class]

        results = []
        batch_size = 1000
        for i in range(0, len(files), batch_size):
            batch_files = files[i:i + batch_size]
            response = requests.post("http://127.0.0.1:8001/detect/", files=batch_files, data=data)
            response.raise_for_status()
            results.extend(response.json())

        return results

    def build_rgb_map_slow(self, rgb_img, depth_img, detect_results: List[dict], pose, step_):
        """ 用观测rgb_img和depth_img更新RGB地图 """
        gap = 2
        resolution_x = 256

        map_pose = (pose[0], -pose[1], -pose[2])
        xyz_points, rgb_points, goods = self.project_rgb_pixels_to_world_coords(
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
    def build_rgb_map(self, rgb_img, depth_img, detect_results: List[dict], pose, step_, vis_info: dict = None):
        """ 用观测rgb_img和depth_img更新RGB地图 """
        gap = 2
        resolution_x = 256

        map_pose = (pose[0], -pose[1], -pose[2])
        xyz_points, rgb_points, goods = self.project_rgb_pixels_to_world_coords(
            rgb_img, depth_img, map_pose, gap=gap, FOV=90, cx=128, cy=128,
            resolution_x=resolution_x, resolution_y=256)

        for detect_result in detect_results:
            center = [int((detect_result['xyxy'][0] + detect_result['xyxy'][2]) / 2),
                      int((detect_result['xyxy'][1] + detect_result['xyxy'][3]) / 2)]
            center = np.array([center])

            center_points, center_rgbs, center_goods = self.project_pixels_to_world_coords(center, rgb_img, depth_img,
                                                                                           map_pose, gap=gap, FOV=90,
                                                                                           cx=128, cy=128,
                                                                                           resolution_x=resolution_x,
                                                                                           resolution_y=256)

            # if detect_result['cls'] not in ['picture']:
            #     continue

            self.object_map.append({
                'position': center_points[:3, 0],
                'label': detect_result['cls'],
                'conf': detect_result['conf']
            })

        self.vis_info = vis_info

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
        # self.object_map = self.deduplicate_objects(self.object_map)

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

    def save_final_map(self, ENLARGE_SIZE=1, display_object_classes: list = None):
        """ 保存最终RGB地图并绘制检测到的物体 """
        # self.object_map = deduplicate_objects(self.object_map)

        grid_sum = self.four_dim_grid_sum[self.min_z_coord:self.max_z_coord + 1, 0:self.THRESHOLD_HIGH,
                   self.min_x_coord:self.max_x_coord + 1, :]
        grid_count = self.four_dim_grid_count[self.min_z_coord:self.max_z_coord + 1, 0:self.THRESHOLD_HIGH,
                     self.min_x_coord:self.max_x_coord + 1]
        zxy_grid = np.argmax(grid_count, axis=1)
        L, M = zxy_grid.shape
        rgb_map = np.zeros((L, M, 3), dtype=np.uint8)

        # slow version
        # for i in range(L):
        #     for j in range(M):
        #         y_idx = zxy_grid[i, j]
        #         count = grid_count[i, y_idx, j]
        #         if count > 0:
        #             rgb = grid_sum[i, y_idx, j] / count
        #             rgb_map[i, j] = np.clip(rgb, 0, 255)

        # fast version
        y_idx = zxy_grid
        count = grid_count[np.arange(L)[:, None], y_idx, np.arange(M)]
        mask = count > 0
        rgb = np.zeros((L, M, 3), dtype=np.float32)
        idx = np.where(mask)
        rgb[idx] = grid_sum[idx[0], y_idx[idx], idx[1]] / count[idx][:, None]
        rgb_map[mask] = np.clip(rgb[mask], 0, 255)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]

        # 放大地图
        rgb_map = cv2.resize(rgb_map, (int(rgb_map.shape[1] * ENLARGE_SIZE), int(rgb_map.shape[0] * ENLARGE_SIZE)),
                             interpolation=cv2.INTER_NEAREST)

        rotated_rgb_map = self.rotate_and_crop_map(rgb_map, ENLARGE_SIZE, crop_size=300)
        self.draw_objects_with_non_overlapping_labels(rotated_rgb_map, ENLARGE_SIZE, display_object_classes, True, True)
        self.draw_waypoints(rotated_rgb_map, ENLARGE_SIZE, True, False)
        plt.imsave(f'{self.saved_folder}/{timestamp}_cropped_rotated_rgb_map.jpg', rotated_rgb_map)

        # plt.imsave(f'{self.saved_folder}/{timestamp}_rgb_map.jpg', rgb_map)

        self.draw_objects_with_non_overlapping_labels(rgb_map, ENLARGE_SIZE, display_object_classes, False, True)
        self.draw_waypoints(rgb_map, ENLARGE_SIZE, False, False)

        plt.imsave(f'{self.saved_folder}/{timestamp}_rgb_map_with_objects.jpg', rgb_map)

        # self._save_complete_rgb_map()

        return f'{self.saved_folder}/{timestamp}_cropped_rotated_rgb_map.jpg'

    def rotate_and_crop_map(self, color_semantic_map, ENLARGE_SIZE, crop_size=100):
        try:
            # 1) 计算agent在当前图中的像素坐标(x, y)
            agent_world = self.vis_info['nodes'][-1]  # 世界坐标(x, y, z)
            map_coords = self.world_to_map_coords(agent_world, False)  # (map_x, map_z, y_layer)

            rel_coords = self.convert_position_from_absolute_to_relative(map_coords[:2])  # (rel_x, rel_z)
            agent_x = int(rel_coords[0] * ENLARGE_SIZE)  # 列
            agent_y = int(rel_coords[1] * ENLARGE_SIZE)  # 行（注意：图像y轴向下）

            # 2) 以agent为旋转中心，将agent朝向转为“向上”
            # 修复：OpenCV正角度为逆时针，这里使用正的角度即可将朝向对齐到上方
            angle_deg = 180 - float(np.degrees(self.vis_info["agent_angle"]))

            self.cropped_agent_info = {
                'agent_x_in_cropped_rotated_map': crop_size * ENLARGE_SIZE // 2,
                'agent_y_in_cropped_rotated_map': crop_size * ENLARGE_SIZE // 2,
                'agent_angle_in_cropped_rotated_map': 0.0,  # 始终向上
                'agent_x_in_original_map': agent_x,
                'agent_y_in_original_map': agent_y,
                'rotation_angle_deg': angle_deg,
                'crop_size': crop_size * ENLARGE_SIZE,
                'ENLARGE_SIZE': ENLARGE_SIZE,
            }

            # angle_deg = float(np.degrees(np.pi / 2))
            M = cv2.getRotationMatrix2D((agent_x, agent_y), angle_deg, 1.0)
            rotated = cv2.warpAffine(
                color_semantic_map,
                M,
                (color_semantic_map.shape[1], color_semantic_map.shape[0]),
                flags=cv2.INTER_NEAREST,  # 避免语义颜色被插值混合
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            # from copy import deepcopy
            # rotated = deepcopy(color_semantic_map)

            # 3) 以agent为中心裁剪(crop_size,crop_size)，自动补边避免越界
            half = crop_size * ENLARGE_SIZE // 2
            h, w = rotated.shape[:2]
            pad_left = max(0, half - agent_x)
            pad_top = max(0, half - agent_y)
            pad_right = max(0, agent_x + half - w)
            pad_bottom = max(0, agent_y + half - h)

            if pad_left or pad_top or pad_right or pad_bottom:
                rotated = cv2.copyMakeBorder(
                    rotated, pad_top, pad_bottom, pad_left, pad_right,
                    borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
                )
                agent_x += pad_left
                agent_y += pad_top

            cropped_rotated_sem_map = rotated[
                                      agent_y - half:agent_y + half,
                                      agent_x - half:agent_x + half
                                      ].copy()
            return cropped_rotated_sem_map
        except Exception as e:
            print(f'旋转与裁剪局部地图失败: {e}')
            print(traceback.format_exc())
        return None

    def draw_objects_with_non_overlapping_labels(self, rgb_map, ENLARGE_SIZE=5, display_object_classes: list = None, rotated: bool = False, location_from_camera: bool = True):
        """在地图上绘制物体，避免标签重叠"""
        object_map = deduplicate_objects(self.object_map, 1.5)
        drawn_labels = []  # 存储已绘制的标签位置和尺寸

        for obj in object_map:
            pos = obj['position']
            label = obj['label']
            conf = obj.get('conf', 1.0)

            if display_object_classes is not None and label not in display_object_classes:
                continue

            # 世界坐标转地图坐标
            # x_map = int((pos[0] - self.min_X) / self.cell_size - self.min_x_coord) * ENLARGE_SIZE
            # z_map = int((self.H - 1 - (pos[2] - self.min_Z) / self.cell_size) - self.min_z_coord) * ENLARGE_SIZE

            x_map, z_map = self.convert_position(pos, ENLARGE_SIZE, rotated, location_from_camera)

            if 0 <= x_map < rgb_map.shape[1] and 0 <= z_map < rgb_map.shape[0]:
                # 绘制圆点
                cv2.circle(rgb_map, (x_map, z_map), 5 * ENLARGE_SIZE, (255, 0, 0), -1)

                # 根据ENLARGE_SIZE缩放文本大小
                # font_scale = max(0.5, ENLARGE_SIZE * 0.4)
                # thickness = max(1, int(ENLARGE_SIZE * 0.7))
                font_scale = ENLARGE_SIZE * 0.4
                thickness = int(ENLARGE_SIZE * 0.7)

                text = f"{label}"
                (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

                # 寻找不重叠的文本位置
                map_agent_pos = self.vis_info['nodes'][-1]
                # 使用与 (x_map, z_map) 一致的坐标系，不做轴互换
                map_agent_pos = self.convert_position(map_agent_pos, ENLARGE_SIZE, rotated, location_from_camera)[:2]

                # 按与 agent 距离最大的方向放置文本（考虑边界与已绘制标签）
                h, w = rgb_map.shape[:2]
                offset = int(ENLARGE_SIZE * 1)
                directions = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
                candidates = []
                for dx, dy in directions:
                    tx = int(x_map + dx * (offset + text_w))
                    ty = int(z_map + dy * (offset + text_h))
                    # 文本以左下角为锚点，确保在图内
                    left = max(0, min(tx, w - text_w))
                    baseline_y = max(text_h, min(ty, h - 1))
                    top = baseline_y - text_h

                    # 检查与已绘制标签是否重叠
                    overlap = False
                    for drawn in drawn_labels:
                        if self._rectangles_overlap(
                                left, top, text_w, text_h,
                                drawn['x'], drawn['y'] - drawn['h'], drawn['w'], drawn['h']
                        ):
                            overlap = True
                            break
                    if overlap:
                        continue

                    # 计算与 agent 的距离（用文本矩形中心）
                    cx = left + text_w / 2.0
                    cy = top + text_h / 2.0
                    dist2 = (cx - map_agent_pos[0]) ** 2 + (cy - map_agent_pos[1]) ** 2
                    candidates.append(((left, baseline_y), dist2))

                # 选择距离最大的候选，否则回退到默认位置
                text_pos = max(candidates, key=lambda it: it[1])[0] if candidates else (x_map, z_map)

                # text_pos = (x_map, z_map)  # 默认位置
                # text_pos = self._find_non_overlapping_position(
                #     x_map, z_map, text_w, text_h, drawn_labels, rgb_map.shape, ENLARGE_SIZE
                # )

                # 绘制文本
                cv2.putText(rgb_map, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, (255, 255, 255), thickness)

                # 记录已绘制的标签区域
                drawn_labels.append({
                    'x': text_pos[0],
                    'y': text_pos[1],
                    'w': text_w,
                    'h': text_h
                })

    def _find_non_overlapping_position(self, center_x, center_y, text_w, text_h, drawn_labels, map_shape, ENLARGE_SIZE):
        """寻找不重叠的文本位置（仅向下偏移）"""
        # 只向下偏移，避免左右偏移
        max_attempts = 10
        for i in range(max_attempts):
            offset_y = (i * (text_h)) * ENLARGE_SIZE
            candidate_x = center_x
            candidate_y = center_y + offset_y

            # 检查是否在地图范围内
            if (candidate_x < 0 or candidate_x + text_w >= map_shape[1] or
                    candidate_y - text_h < 0 or candidate_y >= map_shape[0]):
                continue

            # 检查是否与已有标签重叠
            overlap = False
            for drawn in drawn_labels:
                if self._rectangles_overlap(
                        candidate_x, candidate_y - text_h, text_w, text_h,
                        drawn['x'], drawn['y'] - drawn['h'], drawn['w'], drawn['h']
                ):
                    overlap = True
                    break

            if not overlap:
                return (candidate_x, candidate_y)

        # 如果都重叠，返回默认位置
        return (center_x, center_y + 10 * ENLARGE_SIZE)

    def _rectangles_overlap(self, x1, y1, w1, h1, x2, y2, w2, h2):
        """检查两个矩形是否重叠"""
        return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)

    def _save_complete_rgb_map(self):
        """完整保存RGB地图数据（sum/count裁剪区、元数据与object_map）"""
        try:
            smaller_four_dim_grid_sum = self.four_dim_grid_sum[
                                        self.min_z_coord:self.max_z_coord + 1,
                                        0:self.THRESHOLD_HIGH,
                                        self.min_x_coord:self.max_x_coord + 1,
                                        :
                                        ]
            smaller_four_dim_grid_count = self.four_dim_grid_count[
                                          self.min_z_coord:self.max_z_coord + 1,
                                          0:self.THRESHOLD_HIGH,
                                          self.min_x_coord:self.max_x_coord + 1,
                                          ]

            map_dict = {
                # 坐标边界信息
                'min_x_coord': self.min_x_coord,
                'max_x_coord': self.max_x_coord,
                'min_z_coord': self.min_z_coord,
                'max_z_coord': self.max_z_coord,
                'max_y_coord': self.max_y_coord,

                # 世界坐标系信息
                'min_X': self.min_X,
                'max_X': self.max_X,
                'min_Z': self.min_Z,
                'max_Z': self.max_Z,
                'min_Y': self.min_Y,
                'max_Y': self.max_Y,

                # 地图尺寸和参数
                'W': self.W,
                'H': self.H,
                'cell_size': self.cell_size,
                'THRESHOLD_HIGH': self.THRESHOLD_HIGH,

                # 网格信息
                'x_grid': self.x_grid,
                'z_grid': self.z_grid,
                'y_grid': self.y_grid,

                # 完整的四维网格数据（裁剪后）
                'smaller_four_dim_grid_sum': smaller_four_dim_grid_sum,
                'smaller_four_dim_grid_count': smaller_four_dim_grid_count,

                # 物体地图
                'object_map': self.object_map,

                # 其他参数
                'MIN_DEPTH': self.MIN_DEPTH,
                'MAX_DEPTH': self.MAX_DEPTH,
                'scene_name': self.scene_name,
            }

            np.save(f'{self.saved_folder}/BEV_rgb_map.npy', map_dict)
            print(f'完整RGB地图数据已保存到 {self.saved_folder}/BEV_rgb_map.npy')
        except Exception as e:
            print(f'保存RGB地图数据失败: {e}')

    def load_complete_rgb_map(self, map_file: str):
        """从文件加载完整的RGB地图数据"""
        try:
            map_dict: dict = np.load(map_file, allow_pickle=True).item()

            # 恢复坐标边界信息
            self.min_x_coord = int(map_dict['min_x_coord'])
            self.max_x_coord = int(map_dict['max_x_coord'])
            self.min_z_coord = int(map_dict['min_z_coord'])
            self.max_z_coord = int(map_dict['max_z_coord'])
            self.max_y_coord = int(map_dict['max_y_coord'])

            # 恢复世界坐标系信息
            self.min_X = float(map_dict['min_X'])
            self.max_X = float(map_dict['max_X'])
            self.min_Z = float(map_dict['min_Z'])
            self.max_Z = float(map_dict['max_Z'])
            self.min_Y = float(map_dict['min_Y'])
            self.max_Y = float(map_dict['max_Y'])

            # 恢复地图尺寸和参数
            self.W = int(map_dict['W'])
            self.H = int(map_dict['H'])
            self.cell_size = float(map_dict['cell_size'])
            self.THRESHOLD_HIGH = int(map_dict['THRESHOLD_HIGH'])

            # 恢复网格信息
            self.x_grid = map_dict['x_grid']
            self.z_grid = map_dict['z_grid']
            self.y_grid = map_dict['y_grid']

            # 重新初始化完整的四维网格
            self.four_dim_grid_sum = np.zeros(
                (len(self.z_grid), len(self.y_grid) + 1, len(self.x_grid), 3),
                dtype=np.float32
            )
            self.four_dim_grid_count = np.zeros(
                (len(self.z_grid), len(self.y_grid) + 1, len(self.x_grid)),
                dtype=np.int32
            )

            # 从裁剪区数据恢复到完整网格
            smaller_sum = map_dict['smaller_four_dim_grid_sum']
            smaller_cnt = map_dict['smaller_four_dim_grid_count']

            self.four_dim_grid_sum[
            self.min_z_coord:self.max_z_coord + 1,
            0:self.THRESHOLD_HIGH,
            self.min_x_coord:self.max_x_coord + 1,
            :
            ] = smaller_sum

            self.four_dim_grid_count[
            self.min_z_coord:self.max_z_coord + 1,
            0:self.THRESHOLD_HIGH,
            self.min_x_coord:self.max_x_coord + 1,
            ] = smaller_cnt

            # 恢复物体地图与其他参数
            self.object_map = map_dict['object_map']
            self.MIN_DEPTH = float(map_dict['MIN_DEPTH'])
            self.MAX_DEPTH = float(map_dict['MAX_DEPTH'])
            self.scene_name = map_dict.get('scene_name', self.scene_name)

            # 基于网格重算H/W（以防W/H与网格不一致）
            self.H, self.W = len(self.z_grid), len(self.x_grid)

            print(f'完整RGB地图数据已从 {map_file} 加载成功')
            return True
        except Exception as e:
            print(f'加载RGB地图数据失败: {e}')
            return False

    def world_to_map_coords(self, world_position_3d, location_from_camera):
        """
        将3维世界坐标转换为2维地图坐标

        Args:
            world_x (float): 世界坐标X
            world_y (float): 世界坐标Y (高度)
            world_z (float): 世界坐标Z

        Returns:
            tuple: (map_x, map_z, y_layer) 地图坐标x, z和对应的高度层
            如果坐标超出地图范围，返回 (None, None, None)
        """
        world_x, world_y, world_z = world_position_3d

        # 检查是否在世界坐标范围内
        if (world_x < self.min_X or world_x >= self.max_X or
                world_z < self.min_Z or world_z >= self.max_Z):
            return None, None, None

        # 转换为地图网格坐标
        map_x = int((world_x - self.min_X) / self.cell_size)

        if location_from_camera:
            map_z = (self.H - 1) - int((world_z - self.min_Z) / self.cell_size)
        else:
            map_z = int((world_z - self.min_Z) / self.cell_size)
        y_layer = np.digitize(world_y, self.y_grid)

        # 检查是否在地图网格范围内
        if (map_x < 0 or map_x >= self.W or
                map_z < 0 or map_z >= self.H or
                y_layer < 0 or y_layer > len(self.y_grid)):
            return None, None, None

        return map_x, map_z, y_layer

    def convert_position_from_absolute_to_relative(self, absolute_position_map_2d):
        """
        将绝对地图坐标转换为相对于smaller_four_dim_grid的坐标
        Args:
            absolute_position_map_2d: (map_x, map_z) 绝对地图坐标
        Returns:
            tuple: (rel_x, rel_z) 相对smaller_four_dim_grid的坐标
            如果超出范围，返回(None, None)
        """
        map_x, map_z = absolute_position_map_2d
        rel_x = map_x - self.min_x_coord
        rel_z = map_z - self.min_z_coord
        if rel_x < 0 or rel_x >= (self.max_x_coord - self.min_x_coord + 1):
            return None, None
        if rel_z < 0 or rel_z >= (self.max_z_coord - self.min_z_coord + 1):
            return None, None
        return rel_x, rel_z

    def convert_position_from_relative_to_absolute(self, relative_position_2d):
        """
        将相对于smaller_four_dim_grid的坐标转换为绝对地图坐标
        Args:
            relative_position_2d: (rel_x, rel_z) 相对smaller_four_dim_grid的坐标
        Returns:
            tuple: (map_x, map_z) 绝对地图坐标
            如果超出范围，返回(None, None)
        """
        rel_x, rel_z = relative_position_2d
        map_x = rel_x + self.min_x_coord
        map_z = rel_z + self.min_z_coord
        if map_x < 0 or map_x >= self.W:
            return None, None
        if map_z < 0 or map_z >= self.H:
            return None, None
        return map_x, map_z

    def map_point_after_rotation(self, x, y, center_x, center_y, angle_deg):
        """将(x, y)绕(center_x, center_y)逆时针旋转angle_deg度，返回新坐标"""
        angle_rad = np.deg2rad(-angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        x_shift, y_shift = x - center_x, y - center_y
        x_new = cos_a * x_shift - sin_a * y_shift + center_x
        y_new = sin_a * x_shift + cos_a * y_shift + center_y
        return x_new, y_new

    def map_point_to_cropped_rotated(self, x, y, agent_x, agent_y, angle_deg, crop_size):
        """
        输入：原图坐标(x, y)，旋转中心(agent_x, agent_y)，旋转角度，裁剪尺寸
        输出：在cropped_rotated_sem_map中的坐标
        """
        # 1. 旋转
        x_rot, y_rot = self.map_point_after_rotation(x, y, agent_x, agent_y, angle_deg)
        # 2. 裁剪（以agent为中心，左上角为(agent_x-half, agent_y-half)）
        half = crop_size // 2
        x_crop = x_rot - (agent_x - half)
        y_crop = y_rot - (agent_y - half)
        return int(round(x_crop)), int(round(y_crop))

    def convert_position(self, position, ENLARGE_SIZE, rotated, location_from_camera):
        map_position = self.world_to_map_coords(position, location_from_camera)
        if map_position[0] is None:
            return -1, -1
        x_rel, z_rel = self.convert_position_from_absolute_to_relative(map_position[:2])
        if x_rel is None or z_rel is None:
            return -1, -1

        x_rel = int(x_rel * ENLARGE_SIZE)
        z_rel = int(z_rel * ENLARGE_SIZE)

        if not rotated:
            return x_rel, z_rel

        info = self.cropped_agent_info  # 缓存必存在

        assert ENLARGE_SIZE == info['ENLARGE_SIZE'], "ENLARGE_SIZE与旋转裁剪时不一致"

        px = x_rel
        py = z_rel
        x_rot, y_rot = self.map_point_to_cropped_rotated(
            px, py,
            info['agent_x_in_original_map'],
            info['agent_y_in_original_map'],
            info['rotation_angle_deg'],
            info['crop_size']
        )
        return int(x_rot), int(y_rot)

    def draw_waypoints(self, map, ENLARGE_SIZE=5, rotated: bool = False, location_from_camera: bool = False):
        """在地图上绘制路径点，节点为黄色圆点，ghost为带编号的蓝色圆点，并连接节点和ghost"""
        nodes = self.vis_info['nodes'] if self.vis_info is not None else []
        ghosts = self.vis_info['ghosts'] if self.vis_info is not None else []

        def draw_point_on_map(map, position, ENLARGE_SIZE, color=(0, 255, 255), label=None):
            x_map, z_map = self.convert_position(position, ENLARGE_SIZE, rotated, location_from_camera)

            if 0 <= x_map < map.shape[1] and 0 <= z_map < map.shape[0]:
                thickness = max(1, int(ENLARGE_SIZE * 0.6))
                cv2.circle(map, (x_map, z_map), 6 * ENLARGE_SIZE, color, -1 if label is None else thickness)
                if label is not None:
                    # 缩小字体缩放系数
                    font_scale = max(0.4, ENLARGE_SIZE * 0.22)
                    text_thickness = max(1, int(ENLARGE_SIZE * 0.5))
                    text = str(label)
                    (text_w, text_h), baseline = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
                    )
                    # 让文本居中于圆心
                    text_x = int(x_map - text_w // 2)
                    text_y = int(z_map + text_h // 2)
                    cv2.putText(
                        map, text,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (255, 255, 255), text_thickness
                    )
            else:
                print(f"节点位置超出地图范围: ({x_map}, {z_map})")

        # 绘制节点（黄色圆点）
        for node in nodes[:-1]:  # 不绘制最后一个节点（机器人当前位置）
            draw_point_on_map(map, node, ENLARGE_SIZE, color=(0, 255, 255))

        # 绘制ghost（蓝色圆点+编号）
        for idx, ghost in enumerate(ghosts):
            draw_point_on_map(map, ghost, ENLARGE_SIZE, color=(0, 255, 0), label=idx)

        # 连接每个相邻的node
        for i in range(1, len(nodes)):
            x1, z1 = self.convert_position(nodes[i - 1], ENLARGE_SIZE, rotated, location_from_camera)
            x2, z2 = self.convert_position(nodes[i], ENLARGE_SIZE, rotated, location_from_camera)
            if (0 <= x1 < map.shape[1] and 0 <= z1 < map.shape[0] and
                    0 <= x2 < map.shape[1] and 0 <= z2 < map.shape[0]):
                cv2.line(map, (x1, z1), (x2, z2), (0, 255, 255), 2)

        if len(nodes) > 0:
            from habitat.utils.visualizations import maps as habitat_maps

            map_agent_pos = nodes[-1]
            map_agent_pos = self.convert_position(map_agent_pos, ENLARGE_SIZE, rotated, location_from_camera)[:2][::-1]  # (x, z) -> (z, x)
            habitat_maps.draw_agent(
                image=map,
                agent_center_coord=map_agent_pos,
                agent_rotation=self.vis_info["agent_angle"] if not rotated else np.pi,
                agent_radius_px=min(map.shape[0:2]) // 32,
            )
            # # 使用 OpenCV 绘制带朝向的箭头
            # center_yx = map_agent_pos  # (row, col)
            # center_xy = (int(center_yx[1]), int(center_yx[0]))  # 转为 (x, y)
            # radius_px = max(2, min(map.shape[0:2]) // 32)
            # theta = float(self.vis_info["agent_angle"])
            #
            # # 以“箭头默认朝上(0,-1).”为基准，按逆时针旋转 agent_angle（图像坐标y向下）
            # hx, hy = sin(theta), -cos(theta)
            # head_len = int(radius_px * 0.9)  # 让箭头长度小于圆半径
            # tail_len = int(radius_px * 0.3)
            # tip = (int(center_xy[0] + hx * head_len), int(center_xy[1] + hy * head_len))
            # tail = (int(center_xy[0] - hx * tail_len), int(center_xy[1] - hy * tail_len))

            # thick = max(1, radius_px // 3)
            # cv2.arrowedLine(map, tail, tip, (0, 0, 255), thickness=thick, tipLength=0.35)
            # cv2.circle(map, center_xy, max(1, radius_px // 2), (0, 0, 255), 2)
