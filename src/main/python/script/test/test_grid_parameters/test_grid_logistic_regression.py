# coding: utf-8

import types
import unittest

from grid_parameters import grid_logistic_regression


class TestGridParametersLogisticRegressionTask(unittest.TestCase):

    def test_grid_parameters(self):
        grid = grid_logistic_regression.get_grid_parameters()
        self.assertTrue(isinstance(grid, types.ListType))

        for dic_parameters in grid:
            self.assertTrue(isinstance(dic_parameters, types.DictionaryType))


if __name__ == '__main__':
    unittest.main()