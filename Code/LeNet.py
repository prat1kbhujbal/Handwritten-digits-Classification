import tensorflow as tf


class LeNet(tf.keras.Model):
    """Class for LeNet architecture."""

    def __init__(self, input_shape):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            6,
            kernel_size=5,
            strides=1,
            activation='tanh',
            input_shape=input_shape,
            padding='same')
        self.ap1 = tf.keras.layers.AveragePooling2D()
        self.conv2 = tf.keras.layers.Conv2D(
            16,
            kernel_size=5,
            strides=1,
            activation='tanh',
            padding='valid')
        self.ap2 = tf.keras.layers.AveragePooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(120, activation='tanh')
        self.dense2 = tf.keras.layers.Dense(84, activation='tanh')
        self.dense3 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, input):
        out = self.conv1(input)
        out = self.ap1(out)
        out = self.ap2(out)
        out = self.flatten(out)
        out = self.dense1(out)
        out = self.dense2(out)
        out = self.dense3(out)
        return out


def optimizer(model):
    """Specifying a loss function, an optimizer, and metrics to monitor.

    Args:
        model : model

    Returns:
        _type_: Compiled model
    """
    return model.compile(
        optimizer='adam',
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy'])