
from sklearn.neighbors import KNeighborsClassifier


class KNeighborsClassifierTask:

    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors

    def __str__(self):
        pass

    def define_estimator(self):
        self.estimator = KNeighborsClassifier(n_neighbors=self.n_neighbors)

    def fit(self, X, y):
        self.model = self.estimator.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


if __name__ == "__main__":
    X = [[0], [1], [2], [3]]
    y = [0, 0, 1, 1]
    print("{0}".format("test"))
    neighbors_classifier = KNeighborsClassifierTask(3)
    neighbors_classifier.define_estimator()
    neighbors_classifier.fit(X, y)
    print(neighbors_classifier.predict(X))
