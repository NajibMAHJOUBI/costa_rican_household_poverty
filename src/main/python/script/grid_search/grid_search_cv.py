# coding: utf-8

from classification.classifier_task import ClassifierTask
from classification.decision_tree import DecisionTreeTask
from classification.gaussian_nb import GaussianNBTask
from classification.logistic_regression import LogisticRegressionTask
from classification.nearest_neighbors import KNeighborsClassifierTask
from classification.random_forest import RandomForestTask
from grid_parameters import grid_decision_tree
from grid_parameters import grid_gaussian_nb
from grid_parameters import grid_logistic_regression
from grid_parameters import grid_nearest_neighbors
from grid_parameters import grid_random_forest
from sklearn.model_selection import GridSearchCV


class GridSearchCVTask(ClassifierTask):

    def __init__(self, number_folds, base_estimator, scoring):
        ClassifierTask.__init__(self)
        self.number_folds = number_folds
        self.base_estimator = base_estimator
        self.scoring = scoring

    def __str__(self):
        s = "Number of folds: {0}\n".format(self.number_folds)
        s += "Base estimator: {0}\n".format(self.base_estimator)
        s += "Scoring: {0}\n".format(self.scoring)
        return s

    def define_base_estimator(self):
        classifier = None
        if self.estimator_algorithm == "decision_tree":
            classifier = DecisionTreeTask()
        elif self.estimator_algorithm == "gaussian_nb":
            classifier = GaussianNBTask()
        elif self.estimator_algorithm == "logistic_regression":
            classifier = LogisticRegressionTask()
        elif self.estimator_algorithm == "k_nearest_neighbors":
            classifier = KNeighborsClassifierTask()
        elif self.estimator_algorithm == "random_forest":
            classifier = RandomForestTask()

        classifier.define_estimator()
        return classifier.estimator

    def estimator(self):
        self.estimator = GridSearchCV(estimator=self.define_estimator(),
                                      cv=self.number_folds,
                                      param_grid=self.grid_parameters(),
                                      scoring=self.scoring,
                                      verbose=1)

    def grid_parameters(self):
        if self.estimator_algorithm == "decision_tree":
            return grid_decision_tree.get_grid_parameters()
        elif self.estimator_algorithm == "gaussian_nb":
            return grid_gaussian_nb.get_grid_parameters()
        elif self.estimator_algorithm == "logistic_regression":
            return grid_logistic_regression.get_grid_parameters()
        elif self.estimator_algorithm == "k_nearest_neighbors":
            return grid_nearest_neighbors.get_grid_parameters()
        elif self.estimator_algorithm == "random_forest":
            return grid_random_forest.get_grid_parameters()
