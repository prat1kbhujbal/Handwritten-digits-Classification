import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score


class SVM:
    def __init__(self, train_x, train_y, test_x, test_y, kernel) -> None:
        self.kernel = kernel
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.regularization = [0.5, 0.75, 1]

    def svm(self):
        if self.kernel == "Linear":
            for _c in self.regularization:
                self.linear_svm(_c)
        elif self.kernel == "Polynomial" or self.kernel == "RBF":
            for _c in self.regularization:
                self.kernel_svm(_c)

    def linear_svm(self, _c):
        model = LinearSVC(
            C=_c,
            loss='hinge',
            multi_class="crammer_singer",
            penalty='l2',
            random_state=None)
        model.fit(self.train_x, self.train_y)
        pred_train = model.predict(self.train_x)
        pred_test = model.predict(self.test_x)
        model_acc = accuracy_score(self.train_y, pred_train)
        test_acc = accuracy_score(self.test_y, pred_test)
        print("Model accuracy: ", model_acc)
        print("Test accuracy: ", test_acc)

    def kernel_svm(self, _c):
        if self.kernel == "Polynomial":
            model = SVC(kernel='poly', degree=3, gamma='auto', coef0=1, C=_c)
            model.fit(self.train_x, self.train_y)
            pred_train = model.predict(self.train_x)
            pred_test = model.predict(self.test_x)
            model_acc = accuracy_score(self.train_y, pred_train)
            test_acc = accuracy_score(self.test_y, pred_test)
            print("Training accuracy: ", model_acc)
            print("Test accuracy: ", test_acc)
        else:
            model = SVC(kernel='rbf', gamma=0.5, C=_c)
            model.fit(self.train_x, self.train_y)
            pred_train = model.predict(self.train_x)
            pred_test = model.predict(self.test_x)
            model_acc = accuracy_score(self.train_y, pred_train)
            test_acc = accuracy_score(self.test_y, pred_test)
            print("Model accuracy: ", model_acc)
            print("Test accuracy: ", test_acc)
            