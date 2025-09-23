import cv2
import numpy as np
import matplotlib.pyplot as plt
import heapq
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder


class ToleranceAStarFinder(AStarFinder):
    def __init__(self, tolerance=0, **kwargs):
        super().__init__(**kwargs)
        self.tolerance = tolerance
        self.original_end = None

    def find_path(self, start, end, grid):
        self.original_end = end
        # 如果设置了容差，修改终止条件
        if self.tolerance > 0:
            return self._find_path_with_tolerance(start, end, grid)
        else:
            return super().find_path(start, end, grid)

    def _find_path_with_tolerance(self, start, end, grid):
        """带容差的路径查找"""
        grid.cleanup()
        open_list = []
        start.g = 0
        start.f = 0
        start.opened = True
        heapq.heappush(open_list, (start.f, id(start), start))
        self.runs = 0

        while open_list:
            _, _, node = heapq.heappop(open_list)

            if node.closed:
                continue

            # 检查是否在容差范围内
            distance = abs(node.x - end.x) + abs(node.y - end.y)
            if distance <= self.tolerance:
                # 找到容差范围内的点，构建路径
                path = []
                current = node
                while current:
                    path.append(current)
                    current = current.parent
                return list(reversed(path)), self.runs

            node.closed = True

            # 获取邻居节点
            neighbors = grid.neighbors(node, diagonal_movement=self.diagonal_movement)

            for neighbor in neighbors:
                if neighbor.closed or not neighbor.walkable:
                    continue

                # 计算移动代价
                ng = node.g + self.apply_heuristic(neighbor, node)

                if not neighbor.opened or ng < neighbor.g:
                    neighbor.g = ng
                    neighbor.h = self.apply_heuristic(neighbor, end)
                    neighbor.f = neighbor.g + neighbor.h
                    neighbor.parent = node

                    if not neighbor.opened:
                        neighbor.opened = True
                        heapq.heappush(open_list, (neighbor.f, id(neighbor), neighbor))

            self.runs += 1

        # 没有找到路径
        return [], self.runs


class PathFinder:
    def __init__(self, tolerance=0):
        self.tolerance = tolerance
        self.astar_finder = ToleranceAStarFinder(
            tolerance=tolerance,
            diagonal_movement=DiagonalMovement.always
        )

    def astar(self, start: list, end: list, grid: Grid):
        start_node = grid.node(start[0], start[1])
        end_node = grid.node(end[0], end[1])

        path, runs = self.astar_finder.find_path(start_node, end_node, grid)

        # 优化路径：平滑处理
        if len(path) > 2:
            path = self._smooth_path(path, grid)

        return path, runs

    def _smooth_path(self, path, grid):
        """路径平滑优化"""
        if len(path) <= 2:
            return path

        smoothed_path = [path[0]]  # 保留起点
        i = 0

        while i < len(path) - 1:
            # 从当前点开始，找到最远的可直达点
            farthest_reachable = i + 1

            for j in range(i + 2, len(path)):
                if self._is_line_walkable(path[i], path[j], grid):
                    farthest_reachable = j
                else:
                    break

            # 添加最远可达点
            if farthest_reachable < len(path):
                smoothed_path.append(path[farthest_reachable])

            i = farthest_reachable

        return smoothed_path

    def _is_line_walkable(self, start_node, end_node, grid):
        """检查两点之间的直线是否可通行（使用Bresenham算法）"""
        x0, y0 = start_node.x, start_node.y
        x1, y1 = end_node.x, end_node.y

        # Bresenham直线算法
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0

        while True:
            # 检查当前点是否可通行
            if not grid.node(x, y).walkable:
                return False

            if x == x1 and y == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

        return True

    def visualize_path(self, occ_map, path, start, end):
        """可视化地图和路径"""
        plt.figure(figsize=(12, 8))

        # 显示占用网格地图
        plt.imshow(occ_map, cmap='binary', origin='lower')

        # 绘制路径
        if path:
            path_x = [p.x for p in path]
            path_y = [p.y for p in path]
            plt.plot(path_x, path_y, 'r-', linewidth=2, label='Path')

            # 标记实际到达的终点
            actual_end = path[-1]
            plt.plot(actual_end.x, actual_end.y, 'bo', markersize=10, label='Actual End')

        # 标记起点和目标终点
        plt.plot(start[0], start[1], 'go', markersize=10, label='Start')
        plt.plot(end[0], end[1], 'ro', markersize=10, label='Target End')

        # 如果设置了容差，绘制容差范围
        if self.tolerance > 0:
            circle = plt.Circle((end[0], end[1]), self.tolerance,
                                fill=False, color='orange', linestyle='--',
                                label=f'Tolerance ({self.tolerance})')
            plt.gca().add_patch(circle)

        plt.title('Path Finding with Tolerance Visualization')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


if __name__ == '__main__':
    occ_map = np.load('/home/zhandijia/DockerData/zhandijia-root/ETPNav/tmp/semantic_map/BEV_occ_map_raw.npy')
    occ_map[occ_map != 0] = 1
    occ_map = occ_map[:, :, 0]
    occ_map = cv2.erode(occ_map.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2)

    # 设置容差为5格
    tolerance = 10
    path_finder = PathFinder(tolerance=tolerance)
    grid = Grid(matrix=occ_map)
    start = [59, 164]
    end = [420, 65]
    path, runs = path_finder.astar(start, end, grid)
    print('operations:', runs, 'path length:', len(path))

    if path:
        actual_end = path[-1]
        distance_to_target = abs(actual_end.x - end[0]) + abs(actual_end.y - end[1])
        print(f'实际终点: ({actual_end.x}, {actual_end.y})')
        print(f'距离目标终点: {distance_to_target} 格')

    # 可视化结果
    path_finder.visualize_path(occ_map, path, start, end)
    print('finish')