import argparse
import numpy as np
import tensorflow as tf
from SVM import SVM
from LR import LogisticRegression
from LeNet import LeNet, optimizer, plot
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument(
        '--Method', default='',
        help='classifiers')
    parse.add_argument(
        '--DimRed', default='PCA',
        help='Dimensionality Reduction for SVM and Logistic Regression')
    parse.add_argument(
        '--Kernel', default='Polynomial',
        help='kernel for Kernel SVM')
    args = parse.parse_args()
    method = args.Method
    dim_reduction = args.DimRed
    kernel = args.Kernel

    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
    train_x = train_x / 255.0
    test_x = test_x / 255.0

    if method == "Lenet":
        train_x = tf.expand_dims(train_x, 3)
        test_x = tf.expand_dims(test_x, 3)
        val_x = train_x[:5000]
        val_y = train_y[:5000]
        model = LeNet()
        optimizer(model)
        history = model.fit(
            train_x,
            train_y,
            epochs=10,
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

    elif method == "SVM":
        train_x = train_x.reshape(
            (train_x.shape[0],
             train_x.shape[1] *
             train_x.shape[2]))
        test_x = test_x.reshape(
            (test_x.shape[0],
             test_x.shape[1] *
             test_x.shape[2]))
        scaler = StandardScaler()
        if dim_reduction == "PCA":
            train_x = scaler.fit_transform(train_x)
            test_x = scaler.transform(test_x)
            pca = PCA(n_components=90)
            train_x = pca.fit_transform(train_x)
            test_x = pca.transform(test_x)
        elif dim_reduction == "LDA":
            train_x = scaler.fit_transform(train_x)
            test_x = scaler.transform(test_x)
            lda = LinearDiscriminantAnalysis(n_components=9)
            train_x = lda.fit_transform(train_x, train_y)
            test_x = lda.transform(test_x)
        svm = SVM(train_x, train_y, test_x, test_y, kernel)
        svm.svm()
        svm.plot()
    else:
        train_x = train_x.reshape(
            (train_x.shape[0],
             train_x.shape[1] *
             train_x.shape[2]))
        test_x = test_x.reshape(
            (test_x.shape[0],
             test_x.shape[1] *
             test_x.shape[2]))
        scaler = StandardScaler()
        if dim_reduction == "PCA":
            train_x = scaler.fit_transform(train_x)
            test_x = scaler.transform(test_x)
            pca = PCA(n_components=90)
            train_x = pca.fit_transform(train_x)
            test_x = pca.transform(test_x)
        elif dim_reduction == "LDA":
            train_x = scaler.fit_transform(train_x)
            test_x = scaler.transform(test_x)
            lda = LinearDiscriminantAnalysis(n_components=9)
            train_x = lda.fit_transform(train_x, train_y)
            test_x = lda.transform(test_x)
        LR = LogisticRegression(10, l_r=0.9, epochs=50)
        LR.fit(train_x, train_y)
        training_acc = np.sum(train_y == LR.predict(train_x)) / len(train_y)
        print(f"Training Accuracy : {training_acc:.2f}")
        test_pred = LR.predict(test_x)
        test_acc = np.sum(test_y == test_pred) / len(test_y)
        print(f"Testing Accuracy : {test_acc:.2f}")
        test_acc = np.sum(test_y == test_pred) / len(test_y)
        LR.plot(test_y, test_pred)


if __name__ == '__main__':
    main()
