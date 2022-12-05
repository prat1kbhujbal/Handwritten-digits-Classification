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

    (X_train, train_y), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    print("Performing: ", method)
    if method == "Lenet":
        X_train = tf.expand_dims(X_train, 3)
        X_test = tf.expand_dims(X_test, 3)
        val_x = X_train[:5000]
        val_y = train_y[:5000]
        model = LeNet()
        optimizer(model)
        history = model.fit(
            X_train,
            train_y,
            epochs=10,
            validation_data=(
                val_x,
                val_y))
        print("Evaluation on test data")
        model.evaluate(X_test, y_test)
        y_predicted = model.predict(X_test, verbose=0)
        y_predicted_labels = [np.argmax(i) for i in y_predicted]
        c_m = tf.math.confusion_matrix(
            labels=y_test, predictions=y_predicted_labels)
        plot(c_m, history)

    elif method == "SVM":
        X_train = X_train.reshape(
            (X_train.shape[0],
             X_train.shape[1] *
             X_train.shape[2]))
        X_test = X_test.reshape(
            (X_test.shape[0],
             X_test.shape[1] *
             X_test.shape[2]))
        scaler = StandardScaler()
        print("Dimensionality reduction: ", dim_reduction)
        if dim_reduction == "PCA":
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            pca = PCA(n_components=90)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
        elif dim_reduction == "LDA":
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            lda = LinearDiscriminantAnalysis(n_components=9)
            X_train = lda.fit_transform(X_train, train_y)
            X_test = lda.transform(X_test)
        svm = SVM(X_train, train_y, X_test, y_test, kernel)
        svm.svm()
        svm.plot()
    else:
        X_train = X_train.reshape(
            (X_train.shape[0],
             X_train.shape[1] *
             X_train.shape[2]))
        X_test = X_test.reshape(
            (X_test.shape[0],
             X_test.shape[1] *
             X_test.shape[2]))
        scaler = StandardScaler()
        print("Dimensionality reduction: ",dim_reduction)   
        if dim_reduction == "PCA":
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            pca = PCA(n_components=90)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
        elif dim_reduction == "LDA":
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            lda = LinearDiscriminantAnalysis(n_components=9)
            X_train = lda.fit_transform(X_train, train_y)
            X_test = lda.transform(X_test)
        LR = LogisticRegression(10, l_r=0.9, epochs=100)
        LR.fit(X_train, train_y)
        training_acc = np.sum(train_y == LR.predict(X_train)) / len(train_y)
        print(f"Training Accuracy : {training_acc:.2f}")
        test_pred = LR.predict(X_test)
        test_acc = np.sum(y_test == test_pred) / len(y_test)
        print(f"Testing Accuracy : {test_acc:.2f}")
        test_acc = np.sum(y_test == test_pred) / len(y_test)
        LR.plot(y_test, test_pred)


if __name__ == '__main__':
    main()
