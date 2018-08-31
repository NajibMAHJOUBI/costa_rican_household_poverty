# coding: utf-8

import types
import unittest

from grid_parameters import grid_nearest_neighbors


class TestGridParametersNearestNeighborsTask(unittest.TestCase):

    def test_grid_parameters(self):
        grid = grid_nearest_neighbors.get_grid_parameters()
        self.assertTrue(isinstance(grid, types.ListType))
        self.assertTrue(isinstance(grid[0], types.DictionaryType))

if __name__ == '__main__':
    unittest.main()