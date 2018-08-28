
from sklearn.ensemble import RandomForestClassifier

from classifier_task import ClassifierTask


class RandomForestTask(ClassifierTask):

    def __init__(self):
        ClassifierTask.__init__(self)

    def __str__(self):
        s = "Decision Tree Classifier"
        return s

    def define_estimator(self):
        self.estimator = RandomForestClassifier()


if __name__ == "__main__":
    from sklearn import datasets
    iris = datasets.load_iris()
    X, y = iris.data[:, :2], iris.target

    random_forest = RandomForestTask()
    random_forest.define_estimator()
    random_forest.fit(X, y)
    prediction = random_forest.predict(X)

    assert(prediction.shape[0] == y.shape[0])
