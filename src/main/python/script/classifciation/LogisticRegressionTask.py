
from sklearn.linear_model import LogisticRegression

from ClassifierTask import ClassifierTask


class LogisticRegressionTask(ClassifierTask):

    def __init__(self):
        pass

    def __str__(self):
        pass

    def define_estimator(self):
        self.estimator = LogisticRegression()


if __name__ == "__main__":
    from sklearn import datasets
    iris = datasets.load_iris()
    X, y = iris.data[:, :2], iris.target

    logistic_regression = LogisticRegressionTask()
    logistic_regression.define_estimator()
    logistic_regression.fit(X, y)
    prediction = logistic_regression.predict(X)

    assert (prediction.shape[0] == y.shape[0])

