# coding: utf-8

import unittest

import numpy as np

from features_selector.chi2_selector import Chi2SelectorTask


class TestChi2SelectorTask(unittest.TestCase):

    def test_chi2_selector(self):
        X = np.array([[1,2], [1,3], [0,5], [0,6]])
        y = np.array([[0], [0], [1], [1]])
        selector = Chi2SelectorTask(X, y, 0.05)
        chi2, p_value = selector.chi2_pvalue()

        print(chi2)
        print(p_value)


if __name__ == '__main__':
    unittest.main()