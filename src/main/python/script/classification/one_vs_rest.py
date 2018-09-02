
from sklearn.multiclass import OneVsRestClassifier

from classifier_task import ClassifierTask
from classification.logistic_regression import LogisticRegressionTask
from classification.random_forest import RandomForestTask


class OneVsRestTask(ClassifierTask):

    def __init__(self, classifier):
        ClassifierTask.__init__(self)
        self.__classifier__ = classifier

    def __str__(self):
        s = "OneVsRest Classifier"
        return s

    def local_estimator(self):
        classifier = None
        if self.__classifier__ == "logistic_regression":
            classifier = LogisticRegressionTask()
        elif self.__classifier__ == "random_forest_all_features":
            classifier = RandomForestTask()
        classifier.define_estimator()
        return classifier.get_estimator()

    def define_estimator(self):
        self.estimator = OneVsRestClassifier(self.local_estimator())
