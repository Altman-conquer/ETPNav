import numpy as np
from sklearn.cluster import DBSCAN

import requests
from PIL import Image
import io
from typing import Union, List

ins2cat_dict = {
  "0": "person",
  "1": "bicycle",
  "2": "car",
  "3": "motorcycle",
  "4": "airplane",
  "5": "bus",
  "6": "train",
  "7": "truck",
  "8": "boat",
  "9": "traffic light",
  "10": "fire hydrant",
  "11": "stop sign",
  "12": "parking meter",
  "13": "bench",
  "14": "bird",
  "15": "cat",
  "16": "dog",
  "17": "horse",
  "18": "sheep",
  "19": "cow",
  "20": "elephant",
  "21": "bear",
  "22": "zebra",
  "23": "giraffe",
  "24": "backpack",
  "25": "umbrella",
  "26": "handbag",
  "27": "tie",
  "28": "suitcase",
  "29": "frisbee",
  "30": "skis",
  "31": "snowboard",
  "32": "sports ball",
  "33": "kite",
  "34": "baseball bat",
  "35": "baseball glove",
  "36": "skateboard",
  "37": "surfboard",
  "38": "tennis racket",
  "39": "bottle",
  "40": "wine glass",
  "41": "cup",
  "42": "fork",
  "43": "knife",
  "44": "spoon",
  "45": "bowl",
  "46": "banana",
  "47": "apple",
  "48": "sandwich",
  "49": "orange",
  "50": "broccoli",
  "51": "carrot",
  "52": "hot dog",
  "53": "pizza",
  "54": "donut",
  "55": "cake",
  "56": "chair",
  "57": "couch",
  "58": "potted plant",
  "59": "bed",
  "60": "dining table",
  "61": "toilet",
  "62": "tv",
  "63": "laptop",
  "64": "mouse",
  "65": "remote",
  "66": "keyboard",
  "67": "cell phone",
  "68": "microwave",
  "69": "oven",
  "70": "toaster",
  "71": "sink",
  "72": "refrigerator",
  "73": "book",
  "74": "clock",
  "75": "vase",
  "76": "scissors",
  "77": "teddy bear",
  "78": "hair drier",
  "79": "toothbrush",
  "80": "banner",
  "81": "blanket",
  "82": "bridge",
  "83": "cardboard",
  "84": "counter",
  "85": "curtain",
  "86": "door-stuff",
  "87": "floor-wood",
  "88": "flower",
  "89": "fruit",
  "90": "gravel",
  "91": "house",
  "92": "light",
  "93": "mirror-stuff",
  "94": "net",
  "95": "pillow",
  "96": "platform",
  "97": "playingfield",
  "98": "railroad",
  "99": "river",
  "100": "road",
  "101": "roof",
  "102": "sand",
  "103": "sea",
  "104": "shelf",
  "105": "snow",
  "106": "stairs",
  "107": "tent",
  "108": "towel",
  "109": "wall-brick",
  "110": "wall-stone",
  "111": "wall-tile",
  "112": "wall-wood",
  "113": "water-other",
  "114": "window-blind",
  "115": "window-other",
  "116": "tree-merged",
  "117": "fence-merged",
  "118": "ceiling-merged",
  "119": "sky-other-merged",
  "120": "cabinet-merged",
  "121": "table-merged",
  "122": "floor-other-merged",
  "123": "pavement-merged",
  "124": "mountain-merged",
  "125": "grass-merged",
  "126": "dirt-merged",
  "127": "paper-merged",
  "128": "food-other-merged",
  "129": "building-other-merged",
  "130": "rock-merged",
  "131": "wall-other-merged",
  "132": "rug-merged"
}


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


def apply_color_to_map(semantic_map, dataset='MP3D'):
    """ convert semantic map semantic_map into a colorful visualization color_semantic_map"""
    assert len(semantic_map.shape) == 2
    if dataset == 'MP3D':
        COLOR = d3_41_colors_rgb
        num_classes = 41
    elif dataset == 'HM3D':
        COLOR = colormap(rgb=True)
        num_classes = 300
    elif dataset == "ONEFORMER":
        COLOR = np.array([[i, i, i] for i in range(256)], dtype=np.uint8)
        num_classes = 256
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
        elif dataset == "ONEFORMER":
            color_semantic_map[semantic_map == i] = COLOR[i]
    return color_semantic_map


def init_ins2cat_dict(sim):
    global ins2cat_dict
    scene_semantics = sim.semantic_scene
    ins2cat_dict = {int(obj.id.split("_")[-1]): obj.category.index() for obj in scene_semantics.objects}


def get_ins2cat_dict():
    return ins2cat_dict


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


def get_semantic_segmentation_result(images: list):
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

    def detect(images: List[Union[str, np.ndarray, Image.Image]], server_url="http://127.0.0.1:5000/inference/"):
        files = []
        for idx, img in enumerate(images):
            img_bytes = prepare_image(img)
            files.append(("images", (f"image{idx}.jpg", img_bytes, "image/jpeg")))
        data = []
        # if extra_class:
        #     data = [("extra_class", cls) for cls in extra_class]
        response = requests.post(server_url, files=files, data=data)
        response.raise_for_status()
        return response.json()

    return np.array(detect(images)['results'])
