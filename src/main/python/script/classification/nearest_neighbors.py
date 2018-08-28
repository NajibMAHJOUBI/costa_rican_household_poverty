
from sklearn.neighbors import KNeighborsClassifier

from classifier_task import ClassifierTask


class KNeighborsClassifierTask(ClassifierTask):

    def __init__(self):
        ClassifierTask.__init__(self)

    def __str__(self):
        s = "K Nearest Neighbors Classifier"
        return s

    def define_estimator(self):
        self.estimator = KNeighborsClassifier()


if __name__ == "__main__":
    from sklearn import datasets
    iris = datasets.load_iris()
    X, y = iris.data[:, :2], iris.target

    neighbors_classifier = KNeighborsClassifierTask()
    neighbors_classifier.define_estimator()
    neighbors_classifier.fit(X, y)
    prediction = neighbors_classifier.predict(X)

    assert(prediction.shape[0] == y.shape[0])

    neighbors_classifier.estimator

