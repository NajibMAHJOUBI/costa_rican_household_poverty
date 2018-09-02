# coding: utf-8

import types
import unittest
import sklearn

from grid_parameters import grid_one_vs_rest


class TestGridParametersOneVsRestTask(unittest.TestCase):

    def test_grid_parameters(self):
        grid = grid_one_vs_rest.get_grid_parameters("random_forest")
        self.assertTrue(isinstance(grid, types.ListType))

        for estimator in grid[0]["estimator"]:
            self.assertTrue(isinstance(estimator, sklearn.ensemble.forest.RandomForestClassifier))


if __name__ == '__main__':
    unittest.main()
