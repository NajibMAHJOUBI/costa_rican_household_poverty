# coding: utf-8

import unittest

from classification.svc import SVCTask
from sklearn import datasets


class TestSVCTask(unittest.TestCase):

    def test_svc(self):
        iris = datasets.load_iris()
        X, y = iris.data[:, :2], iris.target
        classifier = SVCTask()
        classifier.define_estimator()
        classifier.fit(X, y)
        prediction = classifier.predict(X)

        self.assertEqual(prediction.shape[0], y.shape[0])


if __name__ == '__main__':
    unittest.main()