import tqdm
import numpy as np
import seaborn as sn
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class LogisticRegression():
    """Class for Logistic Regression."""

    def __init__(self, num_classes, l_r, epochs) -> None:
        self.l_r = l_r
        self.iter = epochs
        self.num_classes = num_classes
        self.loss_l = []
        self.accuracy = []

    def update_grad(self, X, N, sm, y_1):
        w_g = (1 / N) * np.dot(X.T, (sm - y_1))
        b_g = (1 / N) * np.sum(sm - y_1)
        self.w -= self.l_r * w_g
        self.bias -= self.l_r * b_g

    def fit(self, X, y):
        N, n = X.shape
        self.w = np.random.random((n, self.num_classes))
        self.bias = np.random.random(self.num_classes)
        for i in tqdm.tqdm(range(1, self.iter + 1)):
            z = X @ self.w + self.bias
            sf_max = self.softmax(z)
            y_1 = tf.keras.utils.to_categorical(y)
            self.update_grad(X, N, sf_max, y_1)
            self.cross_entropy_loss(y, sf_max)
            accuracy = np.sum(y == self.predict(X)) / len(y)
            self.accuracy.append(accuracy)
            if i % 25 == 0:
                print(
                    f"Epoch : {i} Loss : {self.loss_l[i-1]:.4f} Accuracy : {accuracy:.4f}")
        return self

    def softmax(self, z):
        out = np.exp(z - np.max(z))
        for i in range(len(z)):
            out[i] /= np.sum(out[i])
        return out

    def cross_entropy_loss(self, y, sm):
        loss = -np.mean(np.log(sm[np.arange(len(y)), y]))
        self.loss_l.append(loss)

    def predict(self, X):
        z = X @ self.w + self.bias
        out = self.softmax(z)
        return np.argmax(out, axis=1)

    def plot(self, test_y, test_pred):
        """Function to plot the results

        Args:
            test_y : test labels
            test_pred: test predictions
        """
        plt.figure(1, figsize=(10, 7))
        sn.heatmap(
            confusion_matrix(
                test_y,
                test_pred),
            annot=True,
            fmt='d',
            cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        plt.figure(2, figsize=(12, 9))
        plt.subplot(211)
        plt.plot(np.arange(1, len(self.accuracy) + 1), self.accuracy)
        plt.title('Accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('Epoch')

        plt.subplot(212)
        plt.plot(np.arange(1, self.iter + 1), self.loss_l)
        plt.title('Loss')
        plt.ylabel('loss')
        plt.xlabel('Epoch')
        plt.show()
