import os

from datetime import datetime
from collections import namedtuple

import tensorflow as tf
import tensorflow.keras as k

from Snake.env_renderer import CV2Renderer
from Snake.environment import SnakeMaze
from utils import euclidean_distance

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

from Snake.variables import Status, Direction

timestamp = datetime.now().strftime('%d%h%Y__%H%M%S%f')

record_each = 5
image_size = (122, 122)

max_steps_per_episode = 200
num_episodes = None
gama = .99

epsilon = 1.
epsilon_decay = 0.001
min_epsilon = 0.01
memory_size = 10000

Experience = namedtuple('Experience', ('state', 'direction', 'next_state', 'reward'))

VRew = 0
beta_moving_avg = .98


class ReplayMemory:
    def __init__(self, capacity, ):
        self.experiences = []
        self.capacity = capacity

    def is_full(self):
        return len(self.experiences) >= self.capacity

    def push(self, experience: Experience):
        if self.is_full():
            del self.experiences[np.random.randint(len(self.experiences))]
        self.experiences.append(experience)

    def pop(self):
        return self.experiences.pop(np.random.randint(len(self.experiences)))


class DQN(k.Model):
    """
    AlexNet described in the following paper:
        https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
    TODO: Simpler architecture
    """

    def __init__(self, input_shape,
                 beta=.75, alpha=10e-4, bias=2,
                 dropout_rate=.5):
        super(DQN, self).__init__()
        self.conv1 = k.layers.Conv2D(96, 11, 4, padding='same', activation='relu', input_shape=(*input_shape, 3))
        self.lnr1 = k.layers.Lambda(
            lambda x: tf.nn.local_response_normalization(input=x, beta=beta, alpha=alpha, bias=bias))
        self.pool1 = k.layers.MaxPool2D(pool_size=3, strides=2)

        self.conv2 = k.layers.Conv2D(256, 5, padding='same', activation='relu', )
        self.lnr2 = k.layers.Lambda(
            lambda x: tf.nn.local_response_normalization(input=x, beta=beta, alpha=alpha, bias=bias))
        self.pool2 = k.layers.MaxPool2D(pool_size=3, strides=2)

        self.conv3 = k.layers.Conv2D(384, 3, padding='same', activation='relu')
        self.conv4 = k.layers.Conv2D(384, 5, padding='same', activation='relu')
        self.conv5 = k.layers.Conv2D(256, 5, padding='same', activation='relu')

        self.lat = k.layers.Flatten()
        self.den1 = k.layers.Dense(4096, activation='relu')
        self.drop1 = k.layers.Dropout(dropout_rate)
        self.den2 = k.layers.Dense(4096, activation='relu')
        self.drop2 = k.layers.Dropout(dropout_rate)
        self.out = k.layers.Dense(4)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.lnr1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.lnr2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.lat(x)
        x = self.den1(x)
        x = self.drop1(x)
        x = self.den2(x)
        x = self.drop2(x)
        return self.out(x)


memory = ReplayMemory(memory_size)
optimizer = tf.keras.optimizers.RMSprop()
model = DQN(input_shape=image_size)
model.build(input_shape=(None, *image_size, 3))
model._set_inputs(np.zeros((1, *image_size, 3)))


@tf.function
def train_step(state, target):
    state = tf.reshape(state, (1, *state.shape))
    with tf.GradientTape() as tape:
        Q = model(state, training=True)
        l = loss(Q, target)
    grad = tape.gradient(l, model.trainable_variables)
    optimizer.apply_gradients(zip(grad, model.trainable_variables))


@tf.function
def loss(p, t):
    return tf.reduce_sum(tf.square(t - p))


# TODO: Better parameters
def reward(snake, env: SnakeMaze, direction):
    if snake.status == Status.DEAD:
        return -1000
    else:
        r = 0
        r += env.number_of_steps / 4
        r += 10 if snake.steps_without_food == 1 else 0
        # r -= min([euclidean_distance(snake.body[0], f) for f in env.food])
        # r -= 3 if direction not in [(snake.previous_direction + 1) % 4, (snake.previous_direction - 1) % 4] else 0
        return r


i = 0
epsilon_rolling_rew = []
try:
    while num_episodes is None or i < num_episodes:
        print("____________________________________________________________________________________________")
        is_eval_episode = (i % record_each == 0) or (num_episodes and i == num_episodes - 1)
        env = SnakeMaze(52, 52, 1, with_boundaries=False)
        env.reset()
        renderer = CV2Renderer(env, image_size=image_size)
        tot_rew = 0

        skip_step = False
        for _ in range(max_steps_per_episode):
            if is_eval_episode:
                renderer.record()
            if env.num_active_agents is 0:
                break

            state = renderer.generate_np_img()

            Q = model(np.array([state]))
            direction = np.argmax(Q)

            if np.random.rand(1) < epsilon:
                direction = np.random.randint(4)
                if not memory.is_full():
                    env.step({0: direction})
                    rew = reward(env.snakes[0], env, direction)
                    state2 = renderer.generate_np_img()
                    memory.push(Experience(state, direction, state2, rew))
                    continue

            env.step({0: direction})
            rew = reward(env.snakes[0], env, direction)
            state2 = renderer.generate_np_img()
            tot_rew += rew

            memory.push(Experience(state, direction, state2, rew))

            state, direction, state2, rew = memory.pop()

            targetQ = model(np.array([state2]))

            best_Q = np.max(targetQ)
            targetQ = targetQ.numpy()
            targetQ[0, direction] = rew + gama * best_Q
            targetQ = tf.convert_to_tensor(targetQ)

            train_step(state, targetQ)

        print(f"Episode {i + 1} Done!")
        print(f"Episode reward: {tot_rew}")
        print(f"Epsilon: {epsilon}")
        print(f"Replay Memory size: {len(memory.experiences)}")

        VRew = beta_moving_avg * VRew + (1 - beta_moving_avg) * tot_rew
        epsilon_rolling_rew.append([epsilon, VRew])

        epsilon = epsilon - epsilon_decay

        if epsilon < min_epsilon:
            epsilon = min_epsilon

        if is_eval_episode:
            print()
            renderer.save_images(os.path.join('..', 'game_images', 'simple_dqn', timestamp, f'episode{i}'))
            print(f"Images saved at: {os.path.join('..', 'game_images', 'simple_dqn', timestamp, f'episode{i}')}")
            # renderer.save_video(os.path.join('..', 'game_videos', 'simple_dqn', timestamp, f'episode{i}.avi'))
            # print(f"Video saved at: {os.path.join('..', 'game_videos', 'simple_dqn', timestamp, f'episode{i}.avi')}")
            model.save(os.path.join('..', 'models', 'simple_dqn', timestamp, f'episode{i}'))

            print(f"Model saved at: {os.path.join('..', 'models', 'simple_dqn', timestamp, f'episode{i}')}")

        i += 1
except Exception as e:
    print(e)

epsilon_rolling_rew = np.array(epsilon_rolling_rew)
print(epsilon_rolling_rew)

min_rew = epsilon_rolling_rew[:, 1].min()
epsilon_rolling_rew[:, 1] += 0 - min_rew

max_rew = epsilon_rolling_rew[:, 1].max()
epsilon_rolling_rew[:, 0] = max_rew / epsilon_rolling_rew[:, 0]

filename = os.path.join('..', 'graphs', 'simple_dqn')
os.makedirs(filename, exist_ok=True)
filename = os.path.join(filename, timestamp + '.png')

plt.plot(epsilon_rolling_rew[:, 0], label='Epsilon (Scaled)')
plt.plot(epsilon_rolling_rew[:, 1], label='Reward (Scaled)')
plt.legend()
plt.xlabel("Epoch")
plt.savefig(filename)
