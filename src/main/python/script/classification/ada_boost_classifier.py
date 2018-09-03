
from sklearn.ensemble import AdaBoostClassifier

from classifier_task import ClassifierTask

defaults_parameters = AdaBoostClassifier().get_params()


class AdaBoostClassifierTask(ClassifierTask):

    def __init__(self,
                 base_estimator=defaults_parameters["base_estimators"],
                 n_estimators=defaults_parameters["n_estimators"]):
        ClassifierTask.__init__(self)
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators

    def __str__(self):
        s = "AdaBoost Classifier"
        return s

    def define_estimator(self):
        self.estimator = AdaBoostClassifier(base_estimator=self.base_estimator,
                                            n_estimators=self.n_estimators)


if __name__ == "__main__":
    from sklearn import datasets
    iris = datasets.load_iris()
    X, y = iris.data[:, :2], iris.target

    adda_boost = AdaBoostClassifierTask()
    adda_boost.define_estimator()
    adda_boost.fit(X, y)
    prediction = adda_boost.predict(X)

    assert(prediction.shape[0] == y.shape[0])
