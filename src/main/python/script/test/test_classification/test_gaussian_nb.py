# coding: utf-8

import sys
import unittest

from sklearn import datasets

sys.path.append("/home/mahjoubi/Documents/github/costa_rican_household_poverty/src/main/python/script")
from classification.gaussian_nb import GaussianNBTask


class TestGaussianNBTask(unittest.TestCase):

    def test_gaussian_nb(self):
        iris = datasets.load_iris()
        X, y = iris.data[:, :2], iris.target
        gaussian_nb = GaussianNBTask()
        gaussian_nb.define_estimator()
        gaussian_nb.fit(X, y)
        prediction = gaussian_nb.predict(X)

        self.assertEqual(prediction.shape[0], y.shape[0])


if __name__ == '__main__':
    unittest.main()