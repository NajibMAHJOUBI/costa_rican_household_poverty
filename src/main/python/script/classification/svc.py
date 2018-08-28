
from sklearn.svm import SVC

from classifier_task import ClassifierTask


class SVCTask(ClassifierTask):

    def __init__(self):
        ClassifierTask.__init__(self)

    def __str__(self):
        s = "SVCClassifier"
        return s

    def define_estimator(self):
        self.estimator = SVC()


if __name__ == "__main__":
    from sklearn import datasets
    iris = datasets.load_iris()
    X, y = iris.data[:, :2], iris.target

    svc = SVCTask()
    svc.define_estimator()
    svc.fit(X, y)
    prediction = svc.predict(X)

    assert(prediction.shape[0] == y.shape[0])
