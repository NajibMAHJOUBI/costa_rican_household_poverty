
from sklearn.naive_bayes import GaussianNB

from classifier_task import ClassifierTask


class GaussianNBTask(ClassifierTask):

    def __init__(self):
        ClassifierTask.__init__(self)

    def __str__(self):
        s = "Decision Tree Classifier"
        return s

    def define_estimator(self):
        self.estimator = GaussianNB()


if __name__ == "__main__":
    from sklearn import datasets
    iris = datasets.load_iris()
    X, y = iris.data[:, :2], iris.target

    gaussian_nb = GaussianNBTask()
    gaussian_nb.define_estimator()
    gaussian_nb.fit(X, y)
    prediction = gaussian_nb.predict(X)

    assert(prediction.shape[0] == y.shape[0])
