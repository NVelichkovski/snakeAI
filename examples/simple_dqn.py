import tensorflow as tf
import tensorflow.keras as k

from Snake.environment import SnakeMaze


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


env = SnakeMaze(50, 50, 1)
env.reset()

# TODO: Refactor the environment to keep float32 values
model = DQN(input_shape=env.matrix.shape)
inp_image = env.matrix.reshape((1, *env.matrix.shape, 1)).astype('float32')
print(model(inp_image))
