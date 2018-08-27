
from sklearn.neighbors import KNeighborsClassifier


class KNeighborsClassifierTask:

    def __init__(self):
        pass

    def __str__(self):
        pass

    def define_estimator(self):
        self.estimator = KNeighborsClassifier()

    def fit(self, X, y):
        self.model = self.estimator.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def get_estimator(self):
        return self.estimator


if __name__ == "__main__":
    from sklearn import datasets
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = iris.target

    neighbors_classifier = KNeighborsClassifierTask()
    neighbors_classifier.define_estimator()
    neighbors_classifier.fit(X, y)

    assert(X.shape[0] == y.shape[0])
