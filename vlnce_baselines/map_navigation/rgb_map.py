import io
import os
import pickle
import shutil
from math import pi, tan, cos, sin
from typing import Union, List

import cv2
import numpy as np
import open3d as o3d
import requests
from PIL import Image
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
from matplotlib import pyplot as plt
from numpy import linalg as LA
from sklearn.cluster import DBSCAN


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

        self.pcd = o3d.geometry.PointCloud()

    # object_map: [{'position': (x, y, z), 'label': label, ...}, ...]
    @staticmethod
    def deduplicate_objects(object_map, eps=1.0):
        unique_objects = []
        if not object_map:
            return unique_objects
        positions = np.array([obj['position'] for obj in object_map])
        labels = np.array([obj['label'] for obj in object_map])
        confs = np.array([obj.get('conf', 1.0) for obj in object_map])
        for inst_id in np.unique(labels):
            mask = (labels == inst_id)
            if np.sum(mask) <= 1:
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
                count = len(cluster_points)  # 统计每个聚类中点的个数
                unique_objects.append({'position': center, 'label': inst_id, 'conf': mean_conf, 'count': count})
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

    def project_pixels_to_world_coords(self, pixel_points: np.array, rgb_image, current_depth, current_pose, gap=2, FOV=79,
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
    def build_rgb_map(self, rgb_img, depth_img, detect_results: List[dict], pose, step_):
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

    def save_final_map(self, ENLARGE_SIZE=5):
        """ 保存最终RGB地图并绘制检测到的物体 """
        self.object_map = self.deduplicate_objects(self.object_map)

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

        # 放大地图
        rgb_map = cv2.resize(rgb_map, (int(rgb_map.shape[1] * ENLARGE_SIZE), int(rgb_map.shape[0] * ENLARGE_SIZE)),
                             interpolation=cv2.INTER_NEAREST)

        self.draw_objects_with_non_overlapping_labels(rgb_map, ENLARGE_SIZE)

        plt.imsave(f'{self.saved_folder}/final_rgb_map_with_objects.jpg', rgb_map)

    def draw_objects_with_non_overlapping_labels(self, rgb_map, ENLARGE_SIZE=5):
        """在地图上绘制物体，避免标签重叠"""
        drawn_labels = []  # 存储已绘制的标签位置和尺寸

        for obj in self.object_map:
            pos = obj['position']
            label = obj['label']
            conf = obj.get('conf', 1.0)

            # 世界坐标转地图坐标
            x_map = int((pos[0] - self.min_X) / self.cell_size - self.min_x_coord) * ENLARGE_SIZE
            z_map = int((self.H - 1 - (pos[2] - self.min_Z) / self.cell_size) - self.min_z_coord) * ENLARGE_SIZE

            if 0 <= x_map < rgb_map.shape[1] and 0 <= z_map < rgb_map.shape[0]:
                # 绘制圆点
                cv2.circle(rgb_map, (x_map, z_map), 5 * ENLARGE_SIZE, (255, 0, 0), -1)

                # 计算文本尺寸
                # text = f"{label}({conf:.2f})"
                text = f"{label}"
                font_scale = 2
                # thickness = max(1, int(ENLARGE_SIZE)
                thickness = 2
                (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

                # 寻找不重叠的文本位置
                text_pos = self._find_non_overlapping_position(
                    x_map, z_map, text_w, 10, drawn_labels, rgb_map.shape, ENLARGE_SIZE
                )
                # text_pos = (x_map + 3 * ENLARGE_SIZE, z_map - 3 * ENLARGE_SIZE)

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
