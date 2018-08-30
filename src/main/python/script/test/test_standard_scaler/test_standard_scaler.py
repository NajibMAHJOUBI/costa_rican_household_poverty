# coding: utf-8

import unittest

import numpy as np
from sklearn import datasets

from standard_scaler.standard_scaler import StandardScalerTask


class TestStandardScalerTask(unittest.TestCase):

    def test_standard_scaler(self):
        iris = datasets.load_iris()
        X = iris.data[:, :2]
        scaler = StandardScalerTask(X)
        scaler.define_estimator()
        scaler.fit()
        scaled = scaler.transform()

        self.assertAlmostEqual(np.mean(scaled[:, 0]), 0.0)
        self.assertAlmostEqual(np.std(scaled[:, 0]), 1.0)
        self.assertAlmostEqual(np.mean(scaled[:, 1]), 0.0)
        self.assertAlmostEqual(np.std(scaled[:, 1]), 1.0)

if __name__ == '__main__':
    unittest.main()