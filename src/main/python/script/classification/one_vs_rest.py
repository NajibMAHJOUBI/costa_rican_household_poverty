
from sklearn.multiclass import OneVsRestClassifier

from classifier_task import ClassifierTask


class OneVsRestTask(ClassifierTask):

    def __init__(self, estimator):
        ClassifierTask.__init__(self)
        self.local_estimator = estimator

    def __str__(self):
        s = "OneVsRest Classifier"
        return s

    def define_estimator(self):
        self.estimator = OneVsRestClassifier(estimator=self.local_estimator)
