import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score


class SVM:
    def __init__(self, X_train, y_train, X_test, y_test, kernel) -> None:
        self.kernel = kernel
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.regularization = [0.25, 0.50, 0.75, 1.00, 1.25]
        self.accuracy_list = []

    def svm(self):
        if self.kernel == "Linear":
            print("Kernel ", self.kernel)
            for _c in self.regularization:
                self.linear_svm(_c)
        elif self.kernel == "Polynomial" or self.kernel == "RBF":
            print("Kernel ", self.kernel)
            for _c in self.regularization:
                self.kernel_svm(_c)

    def linear_svm(self, _c):
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
        if self.kernel == "Polynomial":
            model = SVC(kernel='poly', degree=3, gamma='auto', coef0=1, C=_c)
            model.fit(self.X_train, self.y_train)
            pred_train = model.predict(self.X_train)
            pred_test = model.predict(self.X_test)
            model_acc = accuracy_score(self.y_train, pred_train)
            test_acc = accuracy_score(self.y_test, pred_test)
            self.accuracy_list.append(model_acc)
            print(
                f"Regularization : {_c}Test Accuracy : {test_acc:.4f} Training Accuracy : {model_acc:.4f}")

        else:
            model = SVC(kernel='rbf', gamma=0.5, C=_c)
            model.fit(self.X_train, self.y_train)
            pred_train = model.predict(self.X_train)
            pred_test = model.predict(self.X_test)
            model_acc = accuracy_score(self.y_train, pred_train)
            test_acc = accuracy_score(self.y_test, pred_test)
            self.accuracy_list.append(model_acc)
            print(
                f"Regularization : {_c} Test Accuracy : {test_acc:.4f} Training Accuracy : {model_acc:.4f}")

    def plot(self):
        plt.figure(1)
        plt.plot(self.regularization, self.accuracy_list)
        plt.title("Accuracy wrt c")
        plt.xlabel("Regularization")
        plt.ylabel("Accuracy")
        plt.show()
