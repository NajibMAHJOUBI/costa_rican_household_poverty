
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from classifier_task import ClassifierTask


class QuadraticDiscriminantAnalysisTask(ClassifierTask):

    def __init__(self):
        ClassifierTask.__init__(self)

    def __str__(self):
        s = "Quadratic Discriminant Classifier"
        return s

    def define_estimator(self):
        self.estimator = QuadraticDiscriminantAnalysis()


if __name__ == "__main__":
    from sklearn import datasets
    iris = datasets.load_iris()
    X, y = iris.data[:, :2], iris.target

    quadratic_discriminant = QuadraticDiscriminantAnalysisTask()
    quadratic_discriminant.define_estimator()
    quadratic_discriminant.fit(X, y)
    prediction = quadratic_discriminant.predict(X)

    assert(prediction.shape[0] == y.shape[0])
