# coding: utf-8

import unittest

from sklearn import datasets

from over_sampling.over_sampling_task import OverSamplingTask


class TestOverSamplingTask(unittest.TestCase):

    def test_decision_tree(self):
        iris = datasets.load_iris()
        X, y = iris.data[:, :2], iris.target
        over_sampling = OverSamplingTask(X, y)

        x_adasyn, y_adasyn = over_sampling.adasyn()
        self.assertEqual(y_adasyn.shape[0], y_adasyn.shape[0])
        self.assertEqual(type(x_adasyn).__module__,  np.__name__)

        x_smote, y_smote = over_sampling.smote()
        self.assertEqual(x_smote.shape[0], y_smote.shape[0])

# >>> import numpy as np
# a = np.array([1, 2, 3])
# >>> type(a)
# <type 'numpy.ndarray'>
# >>> type(a).__module__
# 'numpy'
# >>> type(a).__module__ == np.__name__


if __name__ == '__main__':
    unittest.main()