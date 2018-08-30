# coding: utf-8

import pandas as pd

from classification.svc import SVCTask
from grid_parameters.grid_svc import get_grid_parameters
from scores import all_scores


class TrainValidationSVC:

    def __init__(self, X_train, X_validation, y_train, y_validation, save_path):
        self.__X_train__ = X_train
        self.__X_validation__ = X_validation
        self.__y_train__ = y_train
        self.__y_validation__ = y_validation
        self.__self_path__ = save_path

    def __str__(self):
        s = "Train Validation Evaluator"
        return s

    def grid_parameters(self, dic_parameters):
        kernel = dic_parameters["kernel"]
        C = dic_parameters["C"]
        degree =  dic_parameters["degree"] if "degree" in dic_parameters.keys() else [3]
        gamma = dic_parameters["gamma"] if "gamma" in dic_parameters.keys() else ["auto"]
        return [(k, c, d, g) for k in kernel for c in C for d in degree for g in gamma]

    def run(self):
        dic_results = {"kernel": [], "C": [], "degree": [], "gamma": [],
                       "accuracy_train": [], "precision_train": [], "recall_train": [], "f1_train": [],
                       "accuracy_validation": [], "precision_validation": [], "recall_validation": [], "f1_validation": []}
        for grid_param in get_grid_parameters():
            for kernel, C, degree, gamma in self.grid_parameters(grid_param):
                print("Kernel: {0}, C: {1}, degree: {2}, gamma: {3}".format(kernel, C, degree, gamma))
                classifier = SVCTask(kernel=kernel, C=C,degree=degree, gamma=gamma)
                classifier.define_estimator()
                classifier.fit(self.__X_train__, self.__y_train__)
                prediction_train = classifier.predict(self.__X_train__)
                prediction_validation = classifier.predict(self.__X_validation__)

                accuracy_train, precision_train, recall_train, f1_train = all_scores.all_scores(self.__y_train__, prediction_train, "macro")
                accuracy_validation, precision_validation, recall_validation, f1_validation = all_scores.all_scores(self.__y_validation__, prediction_validation, "macro")

                dic_results["kernel"].append(kernel)
                dic_results["C"].append(C)
                dic_results["degree"].append(degree)
                dic_results["gamma"].append(gamma)
                dic_results["accuracy_train"].append(accuracy_train)
                dic_results["precision_train"].append(precision_train)
                dic_results["recall_train"].append(recall_train)
                dic_results["f1_train"].append(f1_train)
                dic_results["accuracy_validation"].append(accuracy_validation)
                dic_results["precision_validation"].append(precision_validation)
                dic_results["recall_validation"].append(recall_validation)
                dic_results["f1_validation"].append(f1_validation)

        pd.DataFrame(dic_results).to_csv(self.__self_path__, index=False)