import os

from datetime import datetime
from collections import namedtuple

import tensorflow as tf
import tensorflow.keras as k

from tensorflow.keras.applications.vgg19 import VGG19

from Snake.env_renderer import CV2Renderer
from Snake.environment import SnakeMaze
from utils import euclidean_distance

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

from Snake.variables import Status, Direction

timestamp = datetime.now().strftime('%d%h%Y__%H%M%S%f')

learning_rate = 1e-6

evaluate_each = 50
image_size = (122, 122)

max_steps_per_episode = 200
num_episodes = None
gama = .99

epsilon = 1.
epsilon_decay = 0.0005
min_epsilon = 0.01
memory_size = 10000

Experience = namedtuple('Experience', ('state', 'direction', 'next_state', 'reward'))

VRew = 0
beta_moving_avg = .98

boundaries = True

continue_training = None


class ReplayMemory:
    # TODO: Train in batches
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


memory = ReplayMemory(memory_size)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

if continue_training is None:
    base_model = VGG19(include_top=False, input_shape=(*image_size, 3))
    for l in base_model.layers[:-5]:
        l.trainable = False

    model = k.Sequential([
        base_model,
        k.layers.Flatten(),
        k.layers.Dense(2048, activation='relu', kernel_regularizer='l2'),
        k.layers.Dropout(.5),
        k.layers.Dense(2048, activation='relu', kernel_regularizer='l2'),
        k.layers.Dropout(.5),
        k.layers.Dense(4, activation='softmax', kernel_regularizer='l2'),
    ])
else:
    filename = ['..', 'models', 'transfer_learning_dqn', *continue_training]
    model = tf.keras.models.load_model(os.path.join(*filename))
model.build(input_shape=(None, *image_size, 3))

model.compile(optimizer=optimizer, )


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
        return -1e10
    else:
        r = 0
        r += 1000 if snake.steps_without_food == 1 else 0
        r -= 10 if direction not in [(snake.previous_direction + 1) % 4, (snake.previous_direction - 1) % 4] else 0
        return r


i = 0
epsilon_rolling_rew = []
# try:
while num_episodes is None or i < num_episodes:
    print("____________________________________________________________________________________________")
    is_eval_episode = (i % evaluate_each == 0) or (num_episodes and i == num_episodes - 1)
    env = SnakeMaze(52, 52, 1, with_boundaries=boundaries)
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
                tot_rew += rew
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
        renderer.save_images(os.path.join('..', 'game_images', 'transfer_learning_dqn', timestamp, f'episode{i}'))
        print(
            f"Images saved at: {os.path.join('..', 'game_images', 'transfer_learning_dqn', timestamp, f'episode{i}')}")
        # renderer.save_video(os.path.join('..', 'game_videos', 'transfer_learning_dqn', timestamp, f'episode{i}.avi'))
        # print(f"Video saved at: {os.path.join('..', 'game_videos', 'transfer_learning_dqn', timestamp, f'episode{i}.avi')}")

        try:
            model.save(os.path.join('..', 'models', 'transfer_learning_dqn', timestamp, f'episode{i}'))
            print(f"Model saved at: {os.path.join('..', 'models', 'transfer_learning_dqn', timestamp, f'episode{i}')}")
        except ValueError as e:
            print("Value Error: Error saving the model!")

        epsilon_rolling_rew_ = np.array(epsilon_rolling_rew)

        filename = os.path.join('..', 'graphs', 'transfer_learning_dqn')
        os.makedirs(filename, exist_ok=True)
        filename = os.path.join(filename, timestamp + '.png')

        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(epsilon_rolling_rew_[:, 0], label='Epsilon (Scaled)', color='cornflowerblue')
        ax2.plot(epsilon_rolling_rew_[:, 1], label='Reward (Scaled)', color='brown')

        ax2.set_xlabel("Epoch")

        ax1.set_ylabel("Epsilon")
        ax2.set_ylabel("Average Reward \n(over ~50 samples)")

        plt.tight_layout()
        plt.savefig(filename)

    i += 1
# except Exception as e:
#     print(e)
