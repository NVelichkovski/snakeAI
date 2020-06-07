import tensorflow.keras as k

from Snake.environment import SnakeMaze
from Snake.utils import resize_image, save_images

import matplotlib.pyplot as plt

import numpy as np
import os


def ResNet50(input_shape, num_not_trainable_blocks=4):
    base_model = k.applications.ResNet50(include_top=False, input_shape=input_shape)
    base_model.trainable = False

    for l in base_model.layers:
        if l.name.split("_")[0] == f"conv{num_not_trainable_blocks + 1}":
            break
        l.trainable = True

    model = k.Sequential([
        base_model,
        k.layers.Flatten(),
        k.layers.Dense(4096, activation='relu'),
        k.layers.Dense(2048, activation='relu'),
        k.layers.Dense(1024, activation='relu'),
        k.layers.Dense(4, activation='softmax'),
    ])

    model.build(input_shape=input_shape)
    return model


INPUT_SIZE = (64, 64)
INPUT_SHAPE = (*INPUT_SIZE, 3)

path_to_weights = os.path.join(
    "D:/Google Drive/snakeAI/trainings/transfer_learningDQN\ResNet50/05Jun2020__174051835623/models",
    "episode400")
model = ResNet50(input_shape=INPUT_SHAPE)
model.load_weights(path_to_weights)

env = SnakeMaze(10, 10, 1, with_boundaries=False)
env.reset()
imgs = []

for _ in range(200):
    if env.num_active_agents == 0:
        break
    imgs.append(env.snake_matrices[0].copy())
    # plt.imshow(env.snake_matrices[0].astype(np.uint8))
    # plt.show()
    state = resize_image(env.snake_matrices[0], INPUT_SIZE)
    direction = np.argmax(model(np.array([state])))
    env.step({0: direction})

save_images(imgs, './del_me')