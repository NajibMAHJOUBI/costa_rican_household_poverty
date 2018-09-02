
from sklearn.ensemble import RandomForestClassifier

from classifier_task import ClassifierTask

defaults_parameters = RandomForestClassifier().get_params()


class RandomForestTask(ClassifierTask):

    def __init__(self,
                 n_estimators=defaults_parameters["n_estimators"],
                 criterion=defaults_parameters["criterion"],
                 max_depth=defaults_parameters["max_depth"],
                 min_samples_split=defaults_parameters["min_samples_split"],
                 min_samples_leaf=defaults_parameters["min_samples_leaf"]):
        ClassifierTask.__init__(self)
        self.__n_estimators__ = n_estimators
        self.__criterion__ = criterion
        self.__max_depth__ = max_depth
        self.__min_samples_split__ = min_samples_split
        self.__min_samples_leaf__ = min_samples_leaf

    def __str__(self):
        s = "Random Forest Classifier"
        return s

    def define_estimator(self):
        self.estimator = RandomForestClassifier(n_estimators=self.__n_estimators__,
                                                criterion=self.__criterion__,
                                                max_depth=self.__max_depth__,
                                                min_samples_split=self.__min_samples_split__,
                                                min_samples_leaf=self.__min_samples_leaf__)


if __name__ == "__main__":
    from sklearn import datasets
    iris = datasets.load_iris()
    X, y = iris.data[:, :2], iris.target

    random_forest = RandomForestTask()
    random_forest.define_estimator()
    random_forest.fit(X, y)
    prediction = random_forest.predict(X)

    assert(prediction.shape[0] == y.shape[0])
