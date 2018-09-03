# coding: utf-8

import types
import unittest

import sklearn
from grid_parameters import grid_one_vs_rest


class TestGridParametersOneVsRestTask(unittest.TestCase):

    def test_grid_parameters_rf(self):
        grid = grid_one_vs_rest.get_grid_parameters("random_forest")
        self.assertTrue(isinstance(grid, types.ListType))

        for estimator in grid[0]["estimator"]:
            self.assertTrue(isinstance(estimator, sklearn.ensemble.forest.RandomForestClassifier))

    def test_grid_parameters_dt(self):
        grid = grid_one_vs_rest.get_grid_parameters("decision_tree")
        self.assertTrue(isinstance(grid, types.ListType))

        for estimator in grid[0]["estimator"]:
            self.assertTrue(isinstance(estimator, sklearn.tree.DecisionTreeClassifier))

    def test_grid_parameters_lr(self):
        grid = grid_one_vs_rest.get_grid_parameters("logistic_regression")
        self.assertTrue(isinstance(grid, types.ListType))

        for estimator in grid[0]["estimator"]:
            self.assertTrue(isinstance(estimator, sklearn.linear_model.LogisticRegression))

    def test_grid_parameters_knn(self):
        grid = grid_one_vs_rest.get_grid_parameters("nearest_neighbors", "jaccard")
        self.assertTrue(isinstance(grid, types.ListType))

        for estimator in grid[0]["estimator"]:
            self.assertTrue(isinstance(estimator, sklearn.neighbors.KNeighborsClassifier))

    def test_grid_parameters_svc(self):
        grid = grid_one_vs_rest.get_grid_parameters("svc")
        self.assertTrue(isinstance(grid, types.ListType))

        for estimator in grid[0]["estimator"]:
            self.assertTrue(isinstance(estimator, sklearn.svm.SVC))


if __name__ == '__main__':
    unittest.main()
