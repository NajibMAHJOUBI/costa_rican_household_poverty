
from sklearn.tree import DecisionTreeClassifier

from ClassifierTask import ClassifierTask


class DecisionTreeTask(ClassifierTask):

    def __init__(self):
        pass

    def __str__(self):
        pass

    def define_estimator(self):
        self.estimator = DecisionTreeClassifier()


if __name__ == "__main__":
    from sklearn import datasets
    iris = datasets.load_iris()
    X, y = iris.data[:, :2], iris.target

    decision_tree = DecisionTreeTask()
    decision_tree.define_estimator()
    decision_tree.fit(X, y)
    prediction = decision_tree.predict(X)

    assert(prediction.shape[0] == y.shape[0])

