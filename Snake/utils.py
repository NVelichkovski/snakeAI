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


def save_graph(graphing_data, graph_path):
    plt.style.use('fivethirtyeight')

    fig = plt.figure(figsize=(10, 12),)
    # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, )
    ax = fig.add_subplot(311)
    ax.plot(graphing_data['rolling_avg_epsilon'], label='Epsilon', color='cornflowerblue')
    ax.set_ylabel("Epsilon")

    ax = fig.add_subplot(312)
    ax.plot(graphing_data['rolling_avg_reward'], label='Reward', color='brown')
    ax.set_ylabel("Average Reward \n(over ~50 samples)")

    ax = fig.add_subplot(313)
    ax.plot(graphing_data['Avg Q'], label='Avg Q', color='brown', linewidth=.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Q")

    plt.subplots_adjust(top=1)
    plt.savefig(graph_path + '\\graph.png', )



def save_eval(model, episode_number, episode_images, graphing_data, **kwargs):

    verbose = kwargs['verbose'] if 'verbose' in kwargs else True
    save_images_ = kwargs['save_images'] if 'save_images' in kwargs else False
    training_dir = kwargs['training_dir'] if 'training_dir' in kwargs else './'
    save_videos_ = kwargs['save_videos'] if 'save_videos' in kwargs else False
    save_models_ = kwargs['save_models'] if 'save_models' in kwargs else True
    save_graph_ = kwargs['save_graphs'] if 'save_graphs' in kwargs else True
    override_model = kwargs['override_model'] if 'override_model' in kwargs else True
    

    if save_images_:
        images_path = [training_dir] + ['images', f'episode{episode_number}']
        images_path = os.path.join(*images_path)
        os.makedirs(images_path, exist_ok=True)
        save_images(episode_images, images_path)
        if verbose:
            print(
                f"Images saved at: {os.path.join(*images_path)}")

    if save_videos_:
        video_path = [training_dir] + ['videos', f'episode{episode_number}.mp4']
        os.makedirs(os.path.join(*video_path[:-1]), exist_ok=True)
        video_path = os.path.join(*video_path)
        save_video(episode_images, video_path)
        if verbose:
            print(f"Video saved at:", video_path)

    if save_models_:
        try:
            if override_model:
              models_path = [training_dir] + ['model']  
            else:
              models_path = [training_dir] + ['Model', f'episode{episode_number}']
            models_path = os.path.join(*models_path)
            model.save_weights(models_path)
            if verbose:
                print(f"Model saved at:", models_path)
        except ValueError as e:
            print("Value Error: Error saving the model!")
            print(e)

    if save_graph_:
        os.makedirs(os.path.join(*training_dir), exist_ok=True)
        save_graph(graphing_data, training_dir)
        if verbose:
            print(f"Graphs saved at:", training_dir)
