# coding: utf-8
import sys
sys.path.append("/home/mahjoubi/Documents/github/costa_rican_household_poverty/src/main/python/script")


import unittest
import types
from grid_parameters import grid_decision_tree


class TestGridParametersDecisionTreeTask(unittest.TestCase):

    def test_grid_parameters(self):
        grid = grid_decision_tree.get_grid_parameters()
        self.assertTrue(isinstance(grid, types.ListType))
        parameters_decision_tree = ["criterion", "splitter", "max_depth", "max_features"]

        for dic_parameters in grid:
            for value in dic_parameters.keys():
                self.assertTrue(value in parameters_decision_tree)


if __name__ == '__main__':
    unittest.main()