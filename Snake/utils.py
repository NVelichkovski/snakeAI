import os
from math import sqrt
from typing import Tuple

import cv2
import matplotlib.pyplot as plt

import matplotlib.animation as animation
import numpy as np


from matplotlib import rc
rc('animation', html='jshtml')

from Snake.variables import Direction


def new_position(position, new_direction, matrix):
    i_offset, j_offset = Direction.COORDINATES_OFFSET[new_direction]
    return (position[0] + i_offset) % matrix.shape[0], (position[1] + j_offset) % matrix.shape[1]


def euclidean_distance(point1: Tuple[int, int], point2: Tuple[int, int]):
    return sqrt(sum([(a - b) ** 2 for a, b in zip(point1, point2)]))


def resize_image(img, size=(120, 120), interpolation=cv2.INTER_NEAREST):
    return cv2.resize(img, size, interpolation=interpolation)


def generate_animation(images):
    fig = plt.figure()

    ims = []
    for img in images:
        im = plt.imshow(img, animated=True)
        ims.append([im])

    return animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                     repeat_delay=1000)


def save_images(images, dir_path):
    os.makedirs(dir_path, exist_ok=True)
    for i, img in enumerate(images):
        cv2.imwrite(os.path.join(dir_path, f'{i}.png'), np.array(img))


def save_video(images, filepath):
    raise NotImplementedError


def save_graph(rolling_avg_list, graph_path):
    plt.style.use('fivethirtyeight')

    rolling_avg_list = np.array(rolling_avg_list)
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(rolling_avg_list[:, 0], label='Epsilon', color='cornflowerblue')
    ax2.plot(rolling_avg_list[:, 1], label='Reward', color='brown')

    ax2.set_xlabel("Epoch")

    ax1.set_ylabel("Epsilon")
    ax2.set_ylabel("Average Reward \n(over ~50 samples)")

    plt.tight_layout()
    plt.savefig(graph_path)
