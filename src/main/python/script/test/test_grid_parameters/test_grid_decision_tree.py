# coding: utf-8

import types
import unittest

from grid_parameters import grid_decision_tree
from sklearn.tree import DecisionTreeClassifier


class TestGridParametersDecisionTreeTask(unittest.TestCase):

    def test_grid_parameters(self):
        grid = grid_decision_tree.get_grid_parameters()
        self.assertTrue(isinstance(grid, types.ListType))

        valid_parameters = DecisionTreeClassifier().get_params().keys()
        for dic_parameters in grid:
            self.assertTrue(isinstance(dic_parameters, types.DictionaryType))
            for key in dic_parameters.keys():
                self.assertTrue(key in valid_parameters)


if __name__ == '__main__':
    unittest.main()