# coding: utf-8

import types
import unittest

from grid_parameters import grid_random_forest


class TestGridParametersRandomForestTask(unittest.TestCase):

    def test_grid_parameters(self):
        grid = grid_random_forest.get_grid_parameters()
        self.assertTrue(isinstance(grid, types.ListType))

        for dic_parameters in grid:
            self.assertTrue(isinstance(dic_parameters, types.DictionaryType))


if __name__ == '__main__':
    unittest.main()
