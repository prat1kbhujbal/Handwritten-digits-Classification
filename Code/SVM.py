import sys
import tqdm
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score


class SVM:
    """Class for SVM"""

    def __init__(self, X_train, y_train, X_test, y_test, kernel) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.regularization = [0.25, 0.5, 0.75, 1.0]
        self.accuracy_list = []
        if kernel != "Linear" and kernel != "Polynomial" and kernel != "RBF":
            sys.exit("!!!Please provide valid kernal!!!")
        self.kernel = kernel

    def svm(self):
        print("Kernel: ", self.kernel)
        if self.kernel == "Linear":
            for _c in tqdm.tqdm(self.regularization):
                self.linear_svm(_c)
        elif self.kernel == "Polynomial" or self.kernel == "RBF":
            for _c in tqdm.tqdm(self.regularization):
                self.kernel_svm(_c)

    def linear_svm(self, _c):
        """Function to perform Linear SVM

        Args:
            _c (Float): Regularization parameter
         """
        model = LinearSVC(
            C=_c,
            loss='hinge',
            multi_class="crammer_singer",
            penalty='l2',
            random_state=None)
        model.fit(self.X_train, self.y_train)
        pred_train = model.predict(self.X_train)
        pred_test = model.predict(self.X_test)
        model_acc = accuracy_score(self.y_train, pred_train)
        test_acc = accuracy_score(self.y_test, pred_test)
        self.accuracy_list.append(model_acc)
        print(
            f"Regularization : {_c} Test Accuracy : {test_acc:.4f} Training Accuracy : {model_acc:.4f}")

    def kernel_svm(self, _c):
        """Function to perform kernel SVM

        Args:
            _c (Float): Regularization parameter
         """
        if self.kernel == "Polynomial":
            model = SVC(kernel='poly', degree=2, gamma='auto', coef0=1, C=_c)
            model.fit(self.X_train, self.y_train)
            pred_train = model.predict(self.X_train)
            pred_test = model.predict(self.X_test)
            model_acc = accuracy_score(self.y_train, pred_train)
            test_acc = accuracy_score(self.y_test, pred_test)
            self.accuracy_list.append(model_acc)
            print(
                f"Regularization : {_c} Test Accuracy : {test_acc:.4f} Training Accuracy : {model_acc:.4f}")

        else:
            model = SVC(kernel='rbf', gamma="auto", C=_c)
            model.fit(self.X_train, self.y_train)
            pred_train = model.predict(self.X_train)
            pred_test = model.predict(self.X_test)
            model_acc = accuracy_score(self.y_train, pred_train)
            test_acc = accuracy_score(self.y_test, pred_test)
            self.accuracy_list.append(model_acc)
            print(
                f"Regularization : {_c} Test Accuracy : {test_acc:.4f} Training Accuracy : {model_acc:.4f}")

    def plot(self):
        """Function to plot the results
        """
        plt.figure(1)
        plt.plot(self.regularization, self.accuracy_list)
        plt.title("Accuracy wrt c")
        plt.xlabel("Regularization")
        plt.ylabel("Accuracy")
        plt.show()
