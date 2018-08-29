
from sklearn.neural_network import MLPClassifier

from classifier_task import ClassifierTask


class MLPClassifierTask(ClassifierTask):

    def __init__(self):
        ClassifierTask.__init__(self)

    def __str__(self):
        s = "MLP Classifier"
        return s

    def define_estimator(self):
        self.estimator = MLPClassifier()
