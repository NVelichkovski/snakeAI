import tensorflow as tf
import tensorflow.keras as k

from Snake.environment import SnakeMaze
from models.model_train import train_dqn
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

from Snake.variables import Status

if not tf.test.is_gpu_available():
    print("GPU not found! Training on CPU")


# TODO: Better parameters
def reward(snake, env: SnakeMaze, direction):
    if snake.status == Status.DEAD:
        return -1e6
    else:
        r = 0
        r += env.number_of_steps
        r += 1000 if snake.steps_without_food == 1 else 0
        r -= 10 if direction not in [(snake.previous_direction + 1) % 4, (snake.previous_direction - 1) % 4] else 0
        return r


input_size = (64, 64)
input_shape = (*input_size, 3)
base_model = k.applications.VGG19(include_top=False, input_shape=input_shape)

base_model.trainable = False

model = k.models.Sequential([
    base_model,
    k.layers.Flatten(),
    k.layers.Dense(2304, activation='relu'),
    k.layers.Dense(1152, activation='relu'),
    k.layers.Dense(576, activation='relu'),
    k.layers.Dense(4, activation='softmax'),
])
model.compile()
optimizer = k.optimizers.Adam()


IMAGE_SIZE = (64, 64)
IMAGE_SHAPE = (*IMAGE_SIZE, 3)

learning_rate = 1e-4

config = {
  "save_models": True,
  "save_graphs": True,
  "num_episodes": 1000,
  "gamma": .8,
  "epsilon_decay": 0.005,
  "boundaries": False,
  "maze_width": 10,
  "image_size": IMAGE_SIZE,
}

train_dqn(model, optimizer, reward, **config)
