
from sklearn.multiclass import OneVsRestClassifier

from classifier_task import ClassifierTask
from classification.logistic_regression import LogisticRegressionTask
from classification.random_forest import RandomForestTask


class OneVsRestTask(ClassifierTask):

    def __init__(self):
        ClassifierTask.__init__(self)
        self.__classifier__ = None

    def __str__(self):
        s = "OneVsRest Classifier"
        return s

    def local_estimator(self, type):
        classifier = None
        if type == "logistic_regression":
            classifier = LogisticRegressionTask()
        elif type == "all_features":
            classifier = RandomForestTask()
        classifier.define_estimator()
        self.__classifier__ = classifier.get_estimator()

    def set_classifier(self, classifier):
        self.__classifier__ = classifier

    def define_estimator(self):
        self.estimator = OneVsRestClassifier(estimator=self.__classifier__)
