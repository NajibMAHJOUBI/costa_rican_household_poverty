
from sklearn.linear_model import LogisticRegression

from classifier_task import ClassifierTask


class LogisticRegressionTask(ClassifierTask):

    def __init__(self):
        ClassifierTask.__init__(self)

    def __str__(self):
        s = "Logistic Regression Classifier"
        return s

    def define_estimator(self):
        self.estimator = LogisticRegression()
