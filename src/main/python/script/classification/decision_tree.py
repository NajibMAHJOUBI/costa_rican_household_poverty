
from sklearn.tree import DecisionTreeClassifier

from classifier_task import ClassifierTask


class DecisionTreeTask(ClassifierTask):

    def __init__(self):
        ClassifierTask.__init__(self)

    def __str__(self):
        s = "Decision Tree Classifier"
        return s

    def define_estimator(self):
        self.estimator = DecisionTreeClassifier()
