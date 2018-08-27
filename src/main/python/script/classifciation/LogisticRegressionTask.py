
from sklearn.linear_model import LogisticRegression

class LogisticRegressionTask:

    def __init__(self):
        pass

    def __str__(self):
        pass

    def define_estimator(self):
        self.estimator = LogisticRegression()

    def fit(self, X, y):
        self.model = self.estimator.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


if __name__ == "__main__":
    from sklearn import datasets
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = iris.target

    print("{0}".format("test"))
    logistic_regression = LogisticRegressionTask()
    logistic_regression.define_estimator()
    logistic_regression.fit(X, y)
    print(logistic_regression.predict(X))


    print(type(X))

