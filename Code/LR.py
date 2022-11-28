import numpy as np
import seaborn as sn
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


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
        # onehot_encoder = OneHotEncoder(sparse=False)
        # integer_encoded = y.reshape(len(y), 1)
        # y_hot = onehot_encoder.fit_transform(integer_encoded)
        # y_hot = one_hot(y, c)
        w_grad = (1 / m) * np.dot(X.T, (y_hat - y_hot))
        b_grad = (1 / m) * np.sum(y_hat - y_hot)
        w = w - lr * w_grad
        b = b - lr * b_grad
        loss = -np.mean(np.log(y_hat[np.arange(len(y)), y]))
        losses.append(loss)
        if epoch % 100 == 0:
            pd = predict(X, w, b)
            print('Epoch {epoch}==> Loss = {loss} Acc = {acc}'
                  .format(epoch=epoch, loss=loss, acc=accuracy(y, pd)))
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
    pca = PCA(n_components=100)
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
