import numpy as np
import seaborn as sn
import tensorflow as tf
import matplotlib.pyplot as plt
from LeNet import *


def plot(c_m, history):
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
    plt.legend(['train', 'test'])

    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'])
    plt.show()


def main():
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
    train_x = train_x / 255.0
    test_x = test_x / 255.0
    train_x = tf.expand_dims(train_x, 3)
    test_x = tf.expand_dims(test_x, 3)
    val_x = train_x[:5000]
    val_y = train_y[:5000]
    model = LeNet()
    optimizer(model)
    history = model.fit(
        train_x,
        train_y,
        epochs=30,
        validation_data=(
            val_x,
            val_y))
    print("Evaluation on test data")
    model.evaluate(test_x, test_y)
    y_predicted = model.predict(test_x, verbose=0)
    y_predicted_labels = [np.argmax(i) for i in y_predicted]
    c_m = tf.math.confusion_matrix(
        labels=test_y, predictions=y_predicted_labels)
    plot(c_m, history)


if __name__ == '__main__':
    main()
