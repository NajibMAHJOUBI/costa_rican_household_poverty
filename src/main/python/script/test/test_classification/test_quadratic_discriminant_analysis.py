# coding: utf-8

import unittest

from sklearn import datasets

from classification.quadratic_discriminant_analysis import QuadraticDiscriminantAnalysisTask


class TestQuadraticDiscriminantAnalysisTask(unittest.TestCase):

    def test_nearest_neighbors(self):
        iris = datasets.load_iris()
        X, y = iris.data[:, :2], iris.target
        classifier = QuadraticDiscriminantAnalysisTask()
        classifier.define_estimator()
        classifier.fit(X, y)
        prediction = classifier.predict(X)

        self.assertEqual(prediction.shape[0], y.shape[0])


if __name__ == '__main__':
    unittest.main()