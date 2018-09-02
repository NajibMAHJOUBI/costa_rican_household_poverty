
from sklearn.linear_model import LogisticRegression

from classifier_task import ClassifierTask

defaults_parameters = LogisticRegression().get_params()


class LogisticRegressionTask(ClassifierTask):

    def __init__(self,
                 penalty=defaults_parameters["penalty"],
                 dual=defaults_parameters["dual"],
                 C=defaults_parameters["C"],
                 fit_intercept=defaults_parameters["fit_intercept"]):
        ClassifierTask.__init__(self)
        self.__penalty__ = penalty
        self.__dual__ = dual
        self.__C__ = C
        self.__fit_intercept__ = fit_intercept

    def __str__(self):
        s = "Logistic Regression Classifier"
        return s

    def define_estimator(self):
        self.estimator = LogisticRegression(penalty=self.__penalty__,
                                            dual=self.__dual__,
                                            C=self.__C__,
                                            fit_intercept=self.__fit_intercept__)
