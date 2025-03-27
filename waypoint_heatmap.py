import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import zoom
import cv2
import numpy as np
from scipy.special import softmax
from visualize import visualize_region_attention, visualize_grid_attention, visualize_grid_attention_v2


def run_grid_attention_example(img_path, attention_mask, save_path="test_grid_attention/", version=2,
                               quality=100):
    if attention_mask is None:
        # attention_mask = np.random.randn(64)
        # normed_attention_mask = softmax(attention_mask).reshape(16, 4)
        raise ValueError("Attention mask is None")
    else:
        normed_attention_mask = softmax(attention_mask)

    assert version in [1, 2], "We only support two version of attention visualization example"
    if version == 1:
        visualize_grid_attention(img_path=img_path,
                                 save_path=save_path,
                                 attention_mask=normed_attention_mask,
                                 save_image=True,
                                 save_original_image=True,
                                 quality=quality)
    elif version == 2:
        visualize_grid_attention_v2(img_path=img_path,
                                    save_path=save_path,
                                    attention_mask=normed_attention_mask,
                                    save_image=True,
                                    save_original_image=True,
                                    quality=quality)


# Example probability matrix
NUM_ANGLES = 120
NUM_CLASSES = 12
prob_matrix = np.load('waypoint_prob.npy')
print(prob_matrix.nonzero())
# prob_matrix = np.swapaxes(prob_matrix, 0, 1)
# prob_matrix = np.transpose(prob_matrix)
prob_matrix[prob_matrix != 0] = 1
print(prob_matrix.nonzero())

new_prob_matrix = np.zeros((NUM_CLASSES, NUM_ANGLES))

for i, row in enumerate(prob_matrix):
    for j, col in enumerate(row):
        if prob_matrix[i][j] == 1:
            new_prob_matrix[NUM_CLASSES - j][i] = 1.0
            print(f"{i}, {j}")
print(new_prob_matrix.nonzero())
print(np.transpose(prob_matrix).nonzero())

image = cv2.imread('rgb.png')

run_grid_attention_example('rgb.png', new_prob_matrix)