import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class LogisticRegression():
    def __init__(self, num_classes, l_r=0.9, epochs=1000) -> None:
        self.l_r = l_r
        self.iter = epochs
        self.num_classes = num_classes
        self.loss_arr = []

    def update_grad(self, X, N, sm, y_1):
        w_g = (1 / N) * np.dot(X.T, (sm - y_1))
        b_g = (1 / N) * np.sum(sm - y_1)
        self.w -= self.l_r * w_g
        self.bias -= self.l_r * b_g

    def fit(self, X, y):
        N, n = X.shape
        self.w = np.random.random((n, self.num_classes))
        self.bias = np.random.random(self.num_classes)
        for i in range(self.iter):
            z = X @ self.w + self.bias
            sf_max = self.softmax(z)
            y_1 = tf.keras.utils.to_categorical(y)
            self.update_grad(X, N, sf_max, y_1)
            loss = self.cross_entropy_loss(y, sf_max)
            if i % 100 == 0:
                print('Epoch {epoch}==> Loss = {loss}'
                      .format(epoch=i, loss=loss))
        return self

    def softmax(self, z):
        out = np.exp(z - np.max(z))
        for i in range(len(z)):
            out[i] /= np.sum(out[i])
        return out

    def cross_entropy_loss(self, y, sm):
        loss = -np.mean(np.log(sm[np.arange(len(y)), y]))
        self.loss_arr.append(loss)
        return loss

    def predict(self, X):
        z = X @ self.w + self.bias
        out = self.softmax(z)
        return np.argmax(out, axis=1)


def accuracy(y, y_hat):
    return np.sum(y == y_hat) / len(y)


def main():
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
    train_x = train_x / 255.0
    test_x = test_x / 255.0
    train_x = train_x.reshape(
        (train_x.shape[0],
         train_x.shape[1] *
         train_x.shape[2]))
    test_x = test_x.reshape(
        (test_x.shape[0],
         test_x.shape[1] *
         test_x.shape[2]))
    # scaler = StandardScaler()
    # train_x = scaler.fit_transform(train_x)
    # test_x = scaler.transform(test_x)
    # pca = PCA(n_components=3)
    # train_x = pca.fit_transform(train_x)
    # test_x = pca.transform(test_x)

    LR = LogisticRegression(10)
    LR.fit(train_x, train_y)
    train_preds = LR.predict(train_x)
    tr_acc = accuracy(train_y, train_preds)
    print("tra ", tr_acc)
    test_preds = LR.predict(test_x)
    ts_acc = accuracy(test_y, test_preds)
    print("tsa ", ts_acc)


if __name__ == '__main__':
    main()
