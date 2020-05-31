from Snake.env_renderer import CV2Renderer
from Snake.environment import SnakeMaze

import tensorflow as tf
import numpy as np

import os

filename = ['..', 'models', 'transfer_learning_dqn', '27May2020__232915330322', 'episode4000']

model = tf.keras.models.load_model(os.path.join(*filename))
env = SnakeMaze(50, 50, 1, with_boundaries=True)
env.reset()

renderer = CV2Renderer(env, image_size=(122, 122))
while env.num_active_agents:
    state = renderer.generate_np_img()
    direction = np.argmax(model(np.array([state])))
    env.step({0: direction})
    renderer.render()

renderer.destroy_window()
