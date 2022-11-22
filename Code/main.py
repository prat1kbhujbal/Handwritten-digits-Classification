import tensorflow as tf
from tensorflow import keras
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

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
    keras.layers.AveragePooling2D(), 
    keras.layers.Conv2D(
        16,
        kernel_size=5,
        strides=1,
        activation='tanh',
        padding='valid'),
    keras.layers.AveragePooling2D(),
    keras.layers.Flatten(), 
    keras.layers.Dense(120, activation='tanh'),  
    keras.layers.Dense(84, activation='tanh'), 
    keras.layers.Dense(10, activation='softmax') 
])

lenet_5_model.compile(
    optimizer='adam',
    loss=keras.losses.sparse_categorical_crossentropy,
    metrics=['accuracy'])

lenet_5_model.fit(
    train_x,
    train_y,
    epochs=1,
    validation_data=(
        val_x,
        val_y))

lenet_5_model.evaluate(test_x, test_y)
y_predicted = lenet_5_model.predict(test_x)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
cm = tf.math.confusion_matrix(labels=test_y, predictions=y_predicted_labels)
plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True, fmt='d',cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
