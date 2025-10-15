import cv2
import numpy as np

from vlnce_baselines.map_navigation.map_utils import deduplicate_objects


class map_base_habitat_tools:

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

    def rotate_and_crop_map(self, color_semantic_map, ENLARGE_SIZE, crop_size=100):
        try:
            # 1) 计算agent在当前图中的像素坐标(x, y)
            agent_world = self.vis_info['nodes'][-1]  # 世界坐标(x, y, z)
            map_coords = self.world_to_map_coords(agent_world)  # (map_x, map_z, y_layer)

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
        return None

    def draw_objects_with_non_overlapping_labels(self, rgb_map, ENLARGE_SIZE=5, display_object_classes: list = None, rotated: bool = False):
        """在地图上绘制物体，避免标签重叠"""
        drawn_labels = []  # 存储已绘制的标签位置和尺寸

        object_map = deduplicate_objects(self.object_map, 2) # [obj for obj in self.object_map if obj['label'] == 'table']
        # object_map = self.object_map

        for obj in object_map:
            pos = obj['position']
            label = obj['label']
            conf = obj.get('conf', 1.0)

            if display_object_classes is not None and label not in display_object_classes:
                continue

            # 世界坐标转地图坐标
            # x_map = int((pos[0] - self.min_X) / self.cell_size - self.min_x_coord) * ENLARGE_SIZE
            # z_map = int((self.H - 1 - (pos[2] - self.min_Z) / self.cell_size) - self.min_z_coord) * ENLARGE_SIZE
            x_map, z_map = self.convert_position(pos, ENLARGE_SIZE, rotated)

            if 0 <= x_map < rgb_map.shape[1] and 0 <= z_map < rgb_map.shape[0]:
                # 绘制圆点
                cv2.circle(rgb_map, (x_map, z_map), 5 * ENLARGE_SIZE, (255, 0, 0), -1)

                # 根据ENLARGE_SIZE缩放文本大小
                font_scale = max(0.5, ENLARGE_SIZE * 0.4)
                thickness = max(1, int(ENLARGE_SIZE * 0.7))

                text = f"{label}"
                (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

                # 寻找不重叠的文本位置
                # text_pos = (x_map, z_map)  # 默认位置
                text_pos = self._find_non_overlapping_position(
                    x_map, z_map, text_w, text_h, drawn_labels, rgb_map.shape, ENLARGE_SIZE
                )

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

    def convert_position(self, position, ENLARGE_SIZE, rotated):
        map_position = self.world_to_map_coords(position)
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

    def draw_waypoints(self, map, ENLARGE_SIZE=5, rotated: bool = False):
        """在地图上绘制路径点，节点为黄色圆点，ghost为带编号的蓝色圆点，并连接节点和ghost"""
        nodes = self.vis_info['nodes'] if self.vis_info is not None else []
        ghosts = self.vis_info['ghosts'] if self.vis_info is not None else []

        def draw_point_on_map(map, position, ENLARGE_SIZE, color=(0, 255, 255), label=None):
            x_map, z_map = self.convert_position(position, ENLARGE_SIZE, rotated)

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
            x1, z1 = self.convert_position(nodes[i - 1], ENLARGE_SIZE, rotated)
            x2, z2 = self.convert_position(nodes[i], ENLARGE_SIZE, rotated)
            if (0 <= x1 < map.shape[1] and 0 <= z1 < map.shape[0] and
                    0 <= x2 < map.shape[1] and 0 <= z2 < map.shape[0]):
                cv2.line(map, (x1, z1), (x2, z2), (0, 255, 255), 2)

        if len(nodes) > 0:
            from habitat.utils.visualizations import maps as habitat_maps

            map_agent_pos = nodes[-1]
            map_agent_pos = self.convert_position(map_agent_pos, ENLARGE_SIZE, rotated)[:2][::-1]  # (x, z) -> (z, x)
            habitat_maps.draw_agent(
                image=map,
                agent_center_coord=map_agent_pos,
                agent_rotation=self.vis_info["agent_angle"] if not rotated else np.pi,
                agent_radius_px=min(map.shape[0:2]) // 32,
            )
            # 使用 OpenCV 绘制带朝向的箭头
            center_yx = map_agent_pos  # (row, col)
            center_xy = (int(center_yx[1]), int(center_yx[0]))  # 转为 (x, y)
            radius_px = max(2, min(map.shape[0:2]) // 32)
            theta = float(self.vis_info["agent_angle"])

            # 以“箭头默认朝上(0,-1).”为基准，按逆时针旋转 agent_angle（图像坐标y向下）
            hx, hy = sin(theta), -cos(theta)
            head_len = int(radius_px * 0.9)  # 让箭头长度小于圆半径
            tail_len = int(radius_px * 0.3)
            tip = (int(center_xy[0] + hx * head_len), int(center_xy[1] + hy * head_len))
            tail = (int(center_xy[0] - hx * tail_len), int(center_xy[1] - hy * tail_len))

            thick = max(1, radius_px // 3)
            # cv2.arrowedLine(map, tail, tip, (0, 0, 255), thickness=thick, tipLength=0.35)
            # cv2.circle(map, center_xy, max(1, radius_px // 2), (0, 0, 255), 2)

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

    @staticmethod
    def world_to_map_coords(self, world_position_3d):
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
        map_z = (self.H - 1) - int((world_z - self.min_Z) / self.cell_size)
        y_layer = np.digitize(world_y, self.y_grid)

        # 检查是否在地图网格范围内
        if (map_x < 0 or map_x >= self.W or
                map_z < 0 or map_z >= self.H or
                y_layer < 0 or y_layer > len(self.y_grid)):
            return None, None, None

        return map_x, map_z, y_layer

    @staticmethod
    def map_to_world_coords(self, map_x, map_z, map_y=0):
        """
        将2维地图坐标转换为3维世界坐标

        Args:
            map_x (int): 地图坐标X
            map_z (int): 地图坐标Z
            map_y (int): 高度层索引，默认为0（地面）

        Returns:
            tuple: (world_x, world_y, world_z) 世界坐标
            如果坐标超出范围，返回 (None, None, None)
        """
        # 检查地图坐标是否有效
        if (map_x < 0 or map_x >= self.W or
                map_z < 0 or map_z >= self.H or
                map_y < 0 or map_y >= len(self.y_grid)):
            return None, None, None

        # 转换为世界坐标
        world_x = self.min_X + map_x * self.cell_size + self.cell_size / 2
        world_z = self.min_Z + (self.H - 1 - map_z) * self.cell_size + self.cell_size / 2
        world_y = self.y_grid[map_y] if map_y < len(self.y_grid) else self.y_grid[0]

        return world_x, world_y, world_z

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
