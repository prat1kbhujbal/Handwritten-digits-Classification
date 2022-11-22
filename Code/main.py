from sklearn import metrics
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import os
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
from sklearn.metrics import plot_confusion_matrix

(train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()
train_x = train_x / 255.0
test_x = test_x / 255.0

train_x = tf.expand_dims(train_x, 3)
test_x = tf.expand_dims(test_x, 3)

val_x = train_x[:5000]
val_y = train_y[:5000]

lenet_5_model = keras.models.Sequential([
    keras.layers.Conv2D(
        6,
        kernel_size=5,
        strides=1,
        activation='tanh',
        input_shape=train_x[0].shape,
        padding='same'),
    # C1
    keras.layers.AveragePooling2D(),  # S2
    keras.layers.Conv2D(
        16,
        kernel_size=5,
        strides=1,
        activation='tanh',
        padding='valid'),
    # C3
    keras.layers.AveragePooling2D(),  # S4
    keras.layers.Flatten(),  # Flatten
    keras.layers.Dense(120, activation='tanh'),  # C5
    keras.layers.Dense(84, activation='tanh'),  # F6
    keras.layers.Dense(10, activation='softmax')  # Output layer
])

lenet_5_model.compile(
    optimizer='adam',
    loss=keras.losses.sparse_categorical_crossentropy,
    metrics=['accuracy'])

root_logdir = os.path.join(os.curdir, "logs\\fit\\")


def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
lenet_5_model.fit(
    train_x,
    train_y,
    epochs=1,
    validation_data=(
        val_x,
        val_y),
    callbacks=[tensorboard_cb])
lenet_5_model.evaluate(test_x, test_y)

