# coding: utf-8
import sys
sys.path.append("/home/mahjoubi/Documents/github/costa_rican_household_poverty/src/main/python/script")

from classification.classifier_task import ClassifierTask
from classification.nearest_neighbors import KNeighborsClassifierTask
from classification.decision_tree import DecisionTreeTask
from classification.random_forest import RandomForestTask
from classification.gaussian_nb import GaussianNBTask
from classification.logistic_regression import LogisticRegressionTask
from grid_parameters import grid_decision_tree
from grid_parameters import grid_random_forest
from grid_parameters import grid_nearest_neighbors
from grid_parameters import grid_gaussian_nb
from grid_parameters import grid_logistic_regression
from sklearn.model_selection import GridSearchCV


class GridSearchCVTask(ClassifierTask):

    def __init__(self, number_folds, estimator_algorithm):
        ClassifierTask.__init__(self)
        self.number_folds = number_folds
        self.estimator_algorithm = estimator_algorithm

    def __str__(self):
        s = "Number of folds: {0}\n".format(self.number_folds)
        s += "Estimator algorithm: {0}\n".format(self.estimator_algorithm)
        return s

    def define_estimator(self):
        classifier = None
        if self.estimator_algorithm == "decision_tree":
            classifier = DecisionTreeTask()
        elif self.estimator_algorithm == "gaussian_nb":
            classifier = GaussianNBTask()
        elif self.estimator_algorithm == "logistic_regression":
            classifier = LogisticRegressionTask()
        elif self.estimator_algorithm == "k_nearest_neighbors":
            classifier = KNeighborsClassifierTask()
        elif self.estimator_algorithm == "all_features":
            classifier = RandomForestTask()

        classifier.define_estimator()
        return classifier.estimator

    def grid_search_cv(self):
        self.estimator = GridSearchCV(self.define_estimator(),
                                      cv=self.number_folds,
                                      param_grid=self.grid_parameters(),
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
        elif self.estimator_algorithm == "all_features":
            return grid_random_forest.get_grid_parameters()

    def fit(self, X, y):
        self.model = self.estimator.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


if __name__ == "__main__":
    from sklearn import datasets
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = iris.target

    estimator_algorithms = ["gaussian_nb"]  # "k_nearest_neighbors", "decision_tree", "all_features"
    for estimator_algorithm in estimator_algorithms:
        print("Estimator algorithm: {0}".format(estimator_algorithm))
        search_cv = GridSearchCVTask(3, estimator_algorithm)
        search_cv.grid_search_cv()
        search_cv.fit(X, y)
