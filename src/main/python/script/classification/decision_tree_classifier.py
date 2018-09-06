
from sklearn.tree import DecisionTreeClassifier

from classifier_task import ClassifierTask

defaults_parameters = DecisionTreeClassifier().get_params()


class DecisionTreeClassifierTask(ClassifierTask):

    def __init__(self,
                 criterion=defaults_parameters["criterion"],
                 max_depth=defaults_parameters["max_depth"],
                 min_samples_split=defaults_parameters["min_samples_split"],
                 min_samples_leaf=defaults_parameters["min_samples_leaf"]):
        ClassifierTask.__init__(self)
        self.__criterion__ = criterion
        self.__max_depth__ = max_depth
        self.__min_samples_split__ = min_samples_split
        self.__min_samples_leaf__ = min_samples_leaf

    def __str__(self):
        s = "Decision Tree Classifier"
        return s

    def define_estimator(self):
        self.estimator = DecisionTreeClassifier(criterion=self.__criterion__,
                                                max_depth=self.__max_depth__,
                                                min_samples_split=self.__min_samples_split__,
                                                min_samples_leaf=self.__min_samples_leaf__)
