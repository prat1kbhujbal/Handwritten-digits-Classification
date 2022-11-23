import numpy as np
import seaborn as sn
import tensorflow as tf
import matplotlib.pyplot as plt
from LeNet import *


def main():
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
    train_x = train_x / 255.0
    test_x = test_x / 255.0
    train_x = tf.expand_dims(train_x, 3)
    test_x = tf.expand_dims(test_x, 3)
    val_x = train_x[:5000]
    val_y = train_y[:5000]
    model = LeNet(train_x[0].shape)
    optimizer(model)
    history = model.fit(
        train_x,
        train_y,
        epochs=2,
        validation_data=(
            val_x,
            val_y))
    model.evaluate(test_x, test_y)


if __name__ == '__main__':
    main()
