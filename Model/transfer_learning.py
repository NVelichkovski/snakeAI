import tensorflow as tf
import tensorflow.keras as k


def VGG19(input_shape, num_trainable_layers=5):
    base_model = k.applications.VGG19(include_top=False, input_shape=input_shape)

    base_model.trainable = False
    for layer in base_model.layers[:num_trainable_layers]:
        layer.trainable = True

    model = k.Sequential([
        base_model,
        k.layers.Flatten(),
        k.layers.Dense(2048, activation='relu', kernel_regularizer='l2'),
        k.layers.Dropout(.5),
        k.layers.Dense(2048, activation='relu', kernel_regularizer='l2'),
        k.layers.Dropout(.5),
        k.layers.Dense(4, activation='softmax', kernel_regularizer='l2'),
    ])

    return model