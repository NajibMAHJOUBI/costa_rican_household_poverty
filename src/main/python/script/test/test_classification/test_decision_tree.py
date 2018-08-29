# coding: utf-8
import sys
sys.path.append("/home/mahjoubi/Documents/github/costa_rican_household_poverty/src/main/python/script")

import unittest
from sklearn import datasets
from classification.decision_tree import DecisionTreeTask


class TestDecisionTreeTask(unittest.TestCase):

    def test_decision_tree(self):
        iris = datasets.load_iris()
        X, y = iris.data[:, :2], iris.target
        decision_tree = DecisionTreeTask()
        decision_tree.define_estimator()
        decision_tree.fit(X, y)
        prediction = decision_tree.predict(X)

        self.assertEqual(prediction.shape[0], y.shape[0])


if __name__ == '__main__':
    unittest.main()