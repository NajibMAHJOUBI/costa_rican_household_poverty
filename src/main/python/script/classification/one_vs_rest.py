
from sklearn.multiclass import OneVsRestClassifier

from classifier_task import ClassifierTask
from classification.logistic_regression import LogisticRegressionTask
from classification.random_forest import RandomForestTask


class OneVsRestTask(ClassifierTask):

    def __init__(self):
        ClassifierTask.__init__(self)
        self.__base_estimator__ = None

    def __str__(self):
        s = "OneVsRest Classifier"
        return s

    def base_estimator(self, type):
        classifier = None
        if type == "logistic_regression":
            classifier = LogisticRegressionTask()
        elif type == "random_forest":
            classifier = RandomForestTask()
        classifier.define_estimator()
        self.__base_estimator__ = classifier.get_estimator()

    def set_classifier(self, estimator):
        self.__base_estimator__ = estimator

    def define_estimator(self):
        self.estimator = OneVsRestClassifier(estimator=self.__base_estimator__)
