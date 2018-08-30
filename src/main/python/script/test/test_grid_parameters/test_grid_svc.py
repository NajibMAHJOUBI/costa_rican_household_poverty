# coding: utf-8

import types
import unittest

from grid_parameters import grid_svc


class TestGridParametersGaussianNBTask(unittest.TestCase):

    def test_grid_parameters(self):
        grid = grid_svc.get_grid_parameters()
        self.assertTrue(isinstance(grid, types.ListType))


if __name__ == '__main__':
    unittest.main()
