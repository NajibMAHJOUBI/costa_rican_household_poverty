# coding: utf-8

import unittest

from classification.xgboost_classifier import XGBoostClassifierTask
from sklearn import datasets


class TestXGBoostTask(unittest.TestCase):

    def test_xgboost(self):
        iris = datasets.load_iris()
        X, y = iris.data[:, :2], iris.target
        classifier = XGBoostClassifierTask()
        classifier.define_estimator()
        classifier.fit(X, y)
        prediction = classifier.predict(X)

        self.assertEqual(prediction.shape[0], y.shape[0])


if __name__ == '__main__':
    unittest.main()