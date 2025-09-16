import numpy as np
import matplotlib.pyplot as plt
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder


class PathFinder:
    def __init__(self):
        self.astar_finder = AStarFinder(diagonal_movement=DiagonalMovement.always)

    def astar(self, start: list, end: list, grid: Grid):
        start = grid.node(start[0], start[1])
        end = grid.node(end[0], end[1])

        path, runs = self.astar_finder.find_path(start, end, grid)
        return path, runs

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

        # 标记起点和终点
        plt.plot(start[0], start[1], 'go', markersize=10, label='Start')
        plt.plot(end[0], end[1], 'ro', markersize=10, label='End')

        plt.title('Path Finding Visualization')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


if __name__ == '__main__':
    occ_map = np.load('/home/zhandijia/DockerData/zhandijia-root/ETPNav/tmp/semantic_map/BEV_occ_map_raw.npy')
    occ_map[occ_map != 0] = 1
    occ_map = occ_map[:, :, 0]

    path_finder = PathFinder()
    grid = Grid(matrix=occ_map)
    start = [59, 164]
    end = [420, 65]
    path, runs = path_finder.astar(start, end, grid)
    print('operations:', runs, 'path length:', len(path))

    # 可视化结果
    path_finder.visualize_path(occ_map, path, start, end)
    print('finish')