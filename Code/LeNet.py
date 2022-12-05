"""Lenet architecture"""
import seaborn as sn
import tensorflow as tf
import matplotlib.pyplot as plt


class LeNet(tf.keras.Model):
    """Class for LeNet architecture."""

    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            6,
            kernel_size=5,
            strides=1,
            activation='tanh',
            padding='same')
        self.ap1 = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)
        self.conv2 = tf.keras.layers.Conv2D(
            16,
            kernel_size=5,
            strides=1,
            activation='tanh',
            padding='valid')
        self.ap2 = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(120, activation='tanh')
        self.dense2 = tf.keras.layers.Dense(84, activation='tanh')
        self.dense3 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, input):
        out = self.conv1(input)
        out = self.ap1(out)
        out = self.conv2(out)
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
        Compiled model
    """
    return model.compile(
        optimizer='adam',
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy'])


def plot(c_m, history):
    """Function to plot the results

    Args:
        c_m (dtypr): Representation of confusion matrix
        history : history object
    """
    plt.figure(1, figsize=(10, 7))
    sn.heatmap(c_m, annot=True, fmt='d', cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.figure(2, figsize=(12, 9))
    plt.subplot(211)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'])

    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'])
    plt.show()
