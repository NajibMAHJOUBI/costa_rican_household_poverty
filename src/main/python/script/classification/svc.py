
from sklearn.svm import SVC

from classifier_task import ClassifierTask

defaults_parameters = SVC().get_params()


class SVCTask(ClassifierTask):

    def __init__(self,
                 kernel=defaults_parameters["kernel"],
                 C=defaults_parameters["C"],
                 degree=defaults_parameters["degree"],
                 gamma=defaults_parameters["gamma"]):
        ClassifierTask.__init__(self)
        self.__kernel__ = kernel
        self.__C__ = C
        self.__degree__ = degree
        self.__gamma__ = gamma

    def __str__(self):
        s = "SVC Classifier"
        return s

    def define_estimator(self):
        self.estimator = SVC(C=self.__C__,
                             kernel=self.__kernel__,
                             degree=self.__degree__,
                             gamma=self.__gamma__)


if __name__ == "__main__":
    from sklearn import datasets
    iris = datasets.load_iris()
    X, y = iris.data[:, :2], iris.target

    svc = SVCTask()
    svc.define_estimator()
    svc.fit(X, y)
    prediction = svc.predict(X)

    assert(prediction.shape[0] == y.shape[0])
