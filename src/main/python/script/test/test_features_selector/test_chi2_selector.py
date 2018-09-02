# coding: utf-8

import unittest

import numpy as np
import pandas as pd

from features_selector.chi2_selector import Chi2SelectorTask


class TestChi2SelectorTask(unittest.TestCase):

    def test_chi2_selector(self):
        data = pd.DataFrame({"id": ["a", "b", "c", "d"],
                             "label": [0, 0, 1, 1],
                             "x": [1, 1, 0, 0],
                             "y": [2, 3, 5, 6]})

        selector = Chi2SelectorTask(data, "id", "label", ["x", "y"], 0.05)
        selector.define_restrained_features()

        new_data = selector.get_restrained_features(data, "train")

        self.assertTrue(isinstance(new_data, pd.DataFrame))


if __name__ == '__main__':
    unittest.main()