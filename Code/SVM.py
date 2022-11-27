import numpy as np
import seaborn as sn
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


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
    # train_x = tf.expand_dims(train_x, 3)
    # test_x = tf.expand_dims(test_x, 3)
    train_x = train_x.reshape(
        (train_x.shape[0],
         train_x.shape[1] *
         train_x.shape[2]))
    test_x = test_x.reshape(
        (test_x.shape[0],
         test_x.shape[1] *
         test_x.shape[2]))

    # PCA
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)
    # pca = PCA(n_components=90)
    # train_x = pca.fit_transform(train_x)
    # test_x = pca.transform(test_x)

    # LDA
    lda = LinearDiscriminantAnalysis(n_components=9)
    train_x = lda.fit_transform(train_x, train_y)
    test_x = lda.transform(test_x)
    # Linear
    # Regularization = [0.5, 0.75, 1]
    # for _c in Regularization:
    # model = LinearSVC(
    #     C=0.5,
    #     loss='hinge',
    #     multi_class="crammer_singer",
    #     penalty='l2',
    #     random_state=None)
    # model.fit(train_x, train_y)
    # pred_train = model.predict(train_x)
    # pred_test = model.predict(test_x)
    # model_acc = accuracy_score(train_y, pred_train)
    # test_acc = accuracy_score(test_y, pred_test)
    # print("Model accuracy: ", model_acc)
    # print("Test accuracy: ", test_acc)

    # Poly
    # model = SVC(kernel='poly', degree=3, gamma='auto', coef0=1, C=0.5)
    # model.fit(train_x, train_y)
    # pred_train = model.predict(train_x)
    # pred_test = model.predict(test_x)
    # model_acc = accuracy_score(train_y, pred_train)
    # test_acc = accuracy_score(test_y, pred_test)
    # print("Model accuracy: ", model_acc)
    # print("Test accuracy: ", test_acc)

    # Rad
    model = SVC(kernel='rbf', gamma=0.5, C=0.1)
    model.fit(train_x, train_y)
    pred_train = model.predict(train_x)
    pred_test = model.predict(test_x)
    model_acc = accuracy_score(train_y, pred_train)
    test_acc = accuracy_score(test_y, pred_test)
    print("Model accuracy: ", model_acc)
    print("Test accuracy: ", test_acc)


if __name__ == '__main__':
    main()
