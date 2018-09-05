
from sklearn.ensemble import BaggingClassifier

from classifier_task import ClassifierTask

defaults_parameters = BaggingClassifier().get_params()


class BaggingClassifierTask(ClassifierTask):

    def __init__(self,
                 base_estimator=defaults_parameters["base_estimator"],
                 n_estimators=defaults_parameters["n_estimators"]):
        ClassifierTask.__init__(self)
        self.__base_estimator__ = base_estimator
        self.__n_estimators__ = n_estimators

    def __str__(self):
        s = "Decision Tree Classifier"
        return s

    def define_estimator(self):
        self.estimator = BaggingClassifier(base_estimator=self.__base_estimator__,
                                           n_estimators=self.__n_estimators__)
