# coding: utf-8

import unittest

from classification.decision_tree_classifier import DecisionTreeClassifierTask
from sklearn import datasets


class TestDecisionTreeTask(unittest.TestCase):

    def test_decision_tree(self):
        iris = datasets.load_iris()
        X, y = iris.data[:, :2], iris.target
        classifier = DecisionTreeClassifierTask()
        classifier.define_estimator()
        classifier.fit(X, y)
        prediction = classifier.predict(X)

        self.assertEqual(prediction.shape[0], y.shape[0])


if __name__ == '__main__':
    unittest.main()