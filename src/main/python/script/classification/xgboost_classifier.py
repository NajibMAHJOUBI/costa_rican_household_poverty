# coding: utf-8

from xgboost import XGBClassifier

from classifier_task import ClassifierTask

defaults_parameters = XGBClassifier().get_params()


class XGBoostClassifierTask(ClassifierTask):

    def __init__(self,
                 max_depth=defaults_parameters["max_depth"],
                 n_estimators=defaults_parameters["n_estimators"],
                 learning_rate=defaults_parameters["learning_rate"]):
        ClassifierTask.__init__(self)
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

    def __str__(self):
        s = "XGBoost Classifier"
        return s

    def define_estimator(self):
        self.estimator = XGBClassifier(max_depth=self.max_depth,
                                       n_estimators=self.n_estimators,
                                       learning_rate=self.learning_rate)
