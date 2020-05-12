import tensorflow as tf
import tensorflow.keras as k

from Snake.env_renderer import CV2Renderer
from Snake.environment import SnakeMaze

import numpy as np

from Snake.variables import Status


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
        self.conv1 = k.layers.Conv2D(96, 11, 4, padding='same', activation='relu', input_shape=(*input_shape, 1))
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


optimizer = tf.keras.optimizers.RMSprop()
model = DQN(input_shape=(52, 52))


@tf.function
def train_step(state, target):
    with tf.GradientTape() as tape:
        Q = model(state, training=True)
        l = loss(Q, target)
    grad = tape.gradient(l, model.trainable_variables)
    optimizer.apply_gradients(zip(grad, model.trainable_variables))


@tf.function
def loss(p, t):
    return tf.reduce_sum(tf.square(t - p))


num_step_per_episode = 1000

num_episodes = 100

gama = .99
epsilon = .5


# TODO: Better parameters
def reward(snake):
    return -1000 if snake.status == Status.DEAD else len(snake.body) * 5


cumulative_rewards = []
try:
    for i in range(num_episodes):
        # TODO: Save the model and a game every n-th episodes

        env = SnakeMaze(50, 50, 1)
        env.reset()

        tot_rew = 0
        for _ in range(num_step_per_episode):

            if env.num_active_agents is 0:
                epsilon = 1. / ((i / 50) + 10)
                break

            state = env.matrix.reshape((1, *env.matrix.shape, 1)).astype('float32')
            Q = model(state)
            direction = np.argmax(Q)

            if np.random.rand(1) < epsilon:
                direction = np.random.randint(4)

            env.step({0: direction})
            rew = reward(env.snakes[0])
            tot_rew += rew

            state2 = env.matrix.reshape((1, *env.matrix.shape, 1)).astype('float32')
            targetQ = model(state2)

            best_Q = np.max(targetQ)
            targetQ = targetQ.numpy()
            targetQ[0, direction] = rew + gama * best_Q
            targetQ = tf.convert_to_tensor(targetQ)

            train_step(state, targetQ)

        print(f"Episode {i} Done!\nEpisode reward: {tot_rew}\n")
        cumulative_rewards.append(tot_rew)
except KeyboardInterrupt:
    pass

print(cumulative_rewards)

env = SnakeMaze(50, 50, 1)
env.reset()
renderer = CV2Renderer(env)

while env.num_active_agents:
    renderer.render()

    state = env.matrix.reshape((1, *env.matrix.shape, 1)).astype('float32')
    d = np.argmax(model(state))

    env.step({0: d})
    renderer.render()

renderer.destroy_window()
renderer.save_video('../game_videos')
