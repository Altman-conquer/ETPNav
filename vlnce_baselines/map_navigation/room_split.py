import traceback

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import binary_erosion, binary_dilation, distance_transform_edt
from skimage.morphology import skeletonize
from skimage.feature import peak_local_max
from skimage.measure import label
from skimage.segmentation import watershed
import networkx as nx
from sklearn.neighbors import KDTree


def show_occupancy(matrix, title="Occupancy Map"):
    plt.figure(figsize=(8, 8))
    plt.imshow(matrix, cmap='gray', origin='lower')  # 添加 origin='lower'
    plt.title(title)
    plt.axis('off')
    plt.show()
    # plt.savefig(save_path, bbox_inches='tight', dpi=300)


def width_at_node(n, G, dist_map, resolution):
    """返回该骨架点处的'走廊宽度'（距离变换值×2）"""
    if n not in G.nodes:
        print(f"警告：节点 {n} 不在图中")
        return 0
    node_data = G.nodes[n]
    if 'x' not in node_data or 'y' not in node_data:
        print(f"警告：节点 {n} 缺少坐标属性")
        return 0
    x, y = node_data['x'], node_data['y']
    return dist_map[y, x] * 2 * resolution


def local_width_drop(n, G, dist_map, resolution, ratio=0.7):
    """左右邻居宽度都比它大>ratio → 认为是细脖子"""
    if n not in G.nodes:
        return False

    w0 = width_at_node(n, G, dist_map, resolution)
    if w0 == 0:  # 如果获取宽度失败
        return False

    neighbors = list(G.neighbors(n))
    if len(neighbors) != 2:
        return False
    w1 = width_at_node(neighbors[0], G, dist_map, resolution)
    w2 = width_at_node(neighbors[1], G, dist_map, resolution)
    if w1 == 0 or w2 == 0:  # 如果邻居宽度获取失败
        return False
    return (w0 < w1 * ratio) and (w0 < w2 * ratio)


def room_transition(n, G):
    if n not in G.nodes:
        return False

    nei = list(G.neighbors(n))
    if len(nei) != 2:
        return False

    # 检查所有节点是否存在且有room属性
    if any(neighbor not in G.nodes or 'room' not in G.nodes[neighbor] for neighbor in nei):
        return False
    if 'room' not in G.nodes[n]:
        return False

    r0 = G.nodes[n]['room']
    r1 = G.nodes[nei[0]]['room']
    r2 = G.nodes[nei[1]]['room']
    return (r1 != 0 and r2 != 0 and r1 != r2)

def get_room_label(algorithm_name: str = 'watershed'):
    global skel, dist, labels

    if algorithm_name == 'watershed':
        # 距离变换和分水岭
        dist = distance_transform_edt(free)
        local_max = peak_local_max(
            dist, labels=free.astype(int),
            min_distance=int(0.8 / resolution)
        )
        markers = np.zeros_like(free, dtype=int)
        if len(local_max) > 0:  # 检查是否有峰值
            markers[tuple(local_max.T)] = np.arange(1, len(local_max) + 1)
            labels = watershed(-dist, markers, mask=free)
        else:
            labels = np.zeros_like(free, dtype=int)
            print("警告：未找到房间中心点")
        return labels
    elif algorithm_name == 'flood_fill':
        from skimage.segmentation import flood_fill

        def region_growing_segmentation(free_space, seeds):
            labels = np.zeros_like(free_space, dtype=int)
            for i, (sy, sx) in enumerate(seeds):
                # 使用flood_fill进行区域生长
                mask = flood_fill(free_space, (sy, sx), 1, tolerance=0)
                labels[mask] = i + 1
            return labels

        return region_growing_segmentation(free, [(10, 10), (100, 100)])  # 示例种子点
    elif algorithm_name == 'morphological_geodesic_active_contour':
        from skimage.segmentation import morphological_geodesic_active_contour

        dist = distance_transform_edt(free)
        # 使用距离变换作为初始化
        labels = morphological_geodesic_active_contour(
            dist, num_iter=100, init_level_set='checkerboard'
        )

        return labels

# 主程序
try:
    # 读取数据
    data = np.load('/home/zhandijia/DockerData/zhandijia-root/ETPNav/tmp/semantic_map/BEV_occ_map_raw.npy')
    data = data[:, :, :1]
    print(f"数组形状: {data.shape}")
    print(f"数据类型: {data.dtype}")

    # 二值化处理
    # data = np.where((data == 255 | data == 128), 1, 0)
    data = np.where((data == 255) | (data == 128), 1, 0)
    occupancy = data.squeeze()
    resolution = 0.05

    show_occupancy(occupancy, "Original Occupancy")

    # 形态学处理
    # occupancy = binary_erosion(occupancy, iterations=1)
    #
    # # 找到最大连通区域
    # labeled_regions = label(occupancy, connectivity=2)
    # if labeled_regions.max() > 0:
    #     # 计算每个区域的像素数量
    #     region_sizes = np.bincount(labeled_regions.flat)[1:]  # 排除背景区域0
    #     largest_region_label = np.argmax(region_sizes) + 1
    #     # 只保留最大连通区域
    #     occupancy = (labeled_regions == largest_region_label).astype(int)
    #
    # show_occupancy(occupancy, "Filtered Occupancy")

    # # 填充小空洞
    # from scipy.ndimage import binary_fill_holes

    # 方法1：使用binary_fill_holes填充所有空洞
    # occupancy = binary_fill_holes(occupancy)

    # 方法2：只填充面积小于指定阈值的空洞
    # holes = ~occupancy  # 找到空洞（0区域）
    # hole_labels = label(holes, connectivity=2)
    # min_hole_area = int((1.0 / resolution) ** 2)  # 0.5米×0.5米的空洞阈值
    #
    # for region_id in range(1, hole_labels.max() + 1):
    #     hole_mask = (hole_labels == region_id)
    #     hole_area = np.sum(hole_mask)
    #     if hole_area < min_hole_area:
    #         occupancy[hole_mask] = 1  # 填充小空洞
    #
    # occupancy = occupancy.astype(int)
    #
    # show_occupancy(occupancy, "Hole-filled Occupancy")

    walls = 1 - occupancy
    walls = binary_dilation(walls, iterations=2)
    free = 1 - walls

    labels = get_room_label('watershed')
    unique_labels = np.unique(labels)
    print(f"唯一标签值: {unique_labels}")

    plt.figure(figsize=(10, 8))
    plt.imshow(labels)
    plt.show()


    #
    # # 骨架化
    skel = skeletonize(free)
    show_occupancy(skel, "Skeleton")
    #
    # # 构建骨架图
    # skel_y, skel_x = np.where(skel)
    # if len(skel_x) == 0:
    #     print("错误：未找到骨架点")
    #     exit()
    #
    # skel_labels = labels[skel_y, skel_x]
    #
    # # 构建图结构
    # kdt = KDTree(np.c_[skel_x, skel_y], leaf_size=10)
    # G = nx.Graph()
    #
    # for i, (x, y) in enumerate(zip(skel_x, skel_y)):
    #     G.add_node(i, x=x, y=y, room=skel_labels[i])
    #
    # # 添加边
    # distances, idx = kdt.query_radius(np.c_[skel_x, skel_y], r=1.5, return_distance=True)
    # for i, (nei, d) in enumerate(zip(idx, distances)):
    #     for j, dd in zip(nei, d):
    #         if i != j and dd < 1.5:
    #             G.add_edge(i, j)
    #
    # # 在构建图结构后添加这些调试信息
    # print(f"骨架点数量: {len(skel_x)}")
    # print(f"图中节点数量: {G.number_of_nodes()}")
    # print(f"图中边数量: {G.number_of_edges()}")
    #
    # # 检查节点属性
    # sample_nodes = list(G.nodes())[:5]  # 检查前5个节点
    # for node in sample_nodes:
    #     print(f"节点 {node} 属性: {G.nodes[node]}")
    #
    # # 检查度为2的节点
    # bottleneck_nodes = [n for n in G.nodes if G.degree(n) == 2]
    # print(f"度为2的节点数量: {len(bottleneck_nodes)}")
    # if bottleneck_nodes:
    #     print(f"前几个度为2的节点: {bottleneck_nodes[:5]}")
    # candidates = [n for n in bottleneck_nodes
    #               if local_width_drop(n, G, dist, resolution, ratio=0.65)]
    # door_nodes = [n for n in candidates if room_transition(n, G)]
    #
    # # 输出结果
    # doorway_xy = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in door_nodes]
    # doorway_m = [(x * resolution, y * resolution) for x, y in doorway_xy]
    #
    # print(f"找到 {len(door_nodes)} 个门口")
    # print("门口坐标（像素）:", doorway_xy)
    # print("门口坐标（米）:", doorway_m)
    #
    # # 可视化
    # if doorway_xy:
    #     plt.plot([x for x, y in doorway_xy],
    #              [y for x, y in doorway_xy], 'rX', markersize=10, markeredgewidth=2)
    # plt.title(f'Room Segmentation with {len(door_nodes)} Doorways')
    # plt.colorbar()
    # plt.show()

except FileNotFoundError:
    print("错误：找不到文件 'data/BEV_occ_map_raw.npy'")
except Exception as e:
    print(f"运行错误：{e}")
    print(traceback.format_exc())