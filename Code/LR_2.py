import numpy as np
import tensorflow as tf


class LogisticRegression():
    def __init__(self, l_r=0.01, epochs=1000,num_classes) -> None:
        self.l_r = l_r
        self.iter = epochs
        self.num_classes = num_classes

    def fit(self,X,y):
        
def predict(X, w, b):
    z = X @ w + b
    y_hat = softmax(z)
    return np.argmax(y_hat, axis=1)


def softmax(z):
    exp = np.exp(z - np.max(z))
    for i in range(len(z)):
        exp[i] /= np.sum(exp[i])
    return exp


def fit(X, y, lr, c, epochs):
    m, n = X.shape
    w = np.random.random((n, c))
    b = np.random.random(c)
    losses = []
    for epoch in range(epochs):
        z = X @ w + b
        y_hat = softmax(z)
        y_hot = tf.keras.utils.to_categorical(y)
        w_grad = (1 / m) * np.dot(X.T, (y_hat - y_hot))
        b_grad = (1 / m) * np.sum(y_hat - y_hot)
        w = w - lr * w_grad
        b = b - lr * b_grad
        loss = -np.mean(np.log(y_hat[np.arange(len(y)), y]))
        losses.append(loss)
        if epoch % 100 == 0:
            print('Epoch {epoch}==> Loss = {loss}'
                  .format(epoch=epoch, loss=loss))
    return w, b, losses


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
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)
    pca = PCA(n_components=3)
    train_x = pca.fit_transform(train_x)
    test_x = pca.transform(test_x)

    w, b, l = fit(train_x, train_y, lr=0.9, c=10, epochs=1000)
    train_preds = predict(train_x, w, b)
    tr_acc = accuracy(train_y, train_preds)
    print("tra ", tr_acc)
    test_preds = predict(test_x, w, b)
    ts_acc = accuracy(test_y, test_preds)
    print("tsa ", ts_acc)


if __name__ == '__main__':
    main()
