
from sklearn.ensemble import ExtraTreesClassifier

from classifier_task import ClassifierTask

defaults_parameters = ExtraTreesClassifier().get_params()


class ExtraTreesClassifierTask(ClassifierTask):

    def __init__(self,
                 n_estimators=defaults_parameters["n_estimators"],
                 criterion=defaults_parameters["criterion"],
                 max_depth=defaults_parameters["max_depth"],
                 min_samples_split=defaults_parameters["min_samples_split"],
                 min_samples_leaf=defaults_parameters["min_samples_leaf"]):
        ClassifierTask.__init__(self)
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf=min_samples_leaf
        self.max_depth = max_depth

    def __str__(self):
        s = "ExtraTrees Classifier"
        return s

    def define_estimator(self):
        self.estimator = ExtraTreesClassifier(n_estimators=self.n_estimators,
                                              criterion=self.criterion,
                                              min_samples_split=self.min_samples_split,
                                              min_samples_leaf=self.min_samples_leaf,
                                              max_depth=self.max_depth)
