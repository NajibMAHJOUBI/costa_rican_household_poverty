
from sklearn.ensemble import AdaBoostClassifier

from classifier_task import ClassifierTask


class AdaBoostClassifierTask(ClassifierTask):

    def __init__(self):
        ClassifierTask.__init__(self)

    def __str__(self):
        s = "AdaBoost Classifier"
        return s

    def define_estimator(self):
        self.estimator = AdaBoostClassifier()


if __name__ == "__main__":
    from sklearn import datasets
    iris = datasets.load_iris()
    X, y = iris.data[:, :2], iris.target

    adda_boost = AdaBoostClassifierTask()
    adda_boost.define_estimator()
    adda_boost.fit(X, y)
    prediction = adda_boost.predict(X)

    assert(prediction.shape[0] == y.shape[0])
