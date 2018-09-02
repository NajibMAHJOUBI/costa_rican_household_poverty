
from sklearn.neighbors import KNeighborsClassifier

from classifier_task import ClassifierTask

defaults_parameters = KNeighborsClassifier().get_params()


class KNeighborsClassifierTask(ClassifierTask):

    def __init__(self,
                 n_neighbors=defaults_parameters["n_neighbors"],
                 algorithm=defaults_parameters["algorithm"],
                 weights=defaults_parameters["weights"],
                 leaf_size=defaults_parameters["leaf_size"],
                 metric=defaults_parameters["metric"]):
        ClassifierTask.__init__(self)
        self.__n_neighbors__ = n_neighbors
        self.__algorithm__ = algorithm
        self.__weights__ = weights
        self.__leaf_size__ = leaf_size
        self.__metric__ = metric

    def __str__(self):
        s = "K Nearest Neighbors Classifier"
        return s

    def define_estimator(self):
        self.estimator = KNeighborsClassifier(n_neighbors=self.__n_neighbors__,
                                              algorithm=self.__algorithm__,
                                              weights=self.__weights__,
                                              leaf_size=self.__leaf_size__,
                                              metric=self.__metric__)


if __name__ == "__main__":
    from sklearn import datasets
    iris = datasets.load_iris()
    X, y = iris.data[:, :2], iris.target

    neighbors_classifier = KNeighborsClassifierTask()
    neighbors_classifier.define_estimator()
    neighbors_classifier.fit(X, y)
    prediction = neighbors_classifier.predict(X)

    assert(prediction.shape[0] == y.shape[0])

    neighbors_classifier.estimator

