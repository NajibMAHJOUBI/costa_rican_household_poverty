# coding: utf-8

import os
from itertools import product

from classification.svc import SVCTask
from grid_parameters.grid_svc import get_grid_parameters
from scores import all_scores


class TrainValidationSVC:

    def __init__(self, X_train, X_validation, y_train, y_validation, save_path):
        self.__X_train__ = X_train
        self.__X_validation__ = X_validation
        self.__y_train__ = y_train
        self.__y_validation__ = y_validation
        self.__save_path__ = save_path

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

        file_results = open(os.path.join(self.__save_path__, "results.csv"), "a+")
        file_results.write("index,accuracy,precision,recall,f1\n")

        file_classifier = open(os.path.join(self.__save_path__, "classifier.csv"), "a+")
        file_classifier.write("index,kernel,C,degree,gamma\n")

        index = 0
        for grid_param in get_grid_parameters():
            for kernel, C, degree, gamma in product(grid_param["kernel"],
                                                    grid_param["C"],
                                                    grid_param["degree"],
                                                    grid_param["gamma"]):
                print("{0}, {1}, {2}, {3}, {4}".format(index, kernel, C, degree, gamma))
                classifier = SVCTask(kernel=kernel,
                                     C=C,
                                     degree=degree,
                                     gamma=gamma)
                classifier.define_estimator()
                classifier.fit(self.__X_train__, self.__y_train__)
                prediction = classifier.predict(self.__X_validation__)
                accuracy, precision, recall, f1 = all_scores.all_scores(self.__y_validation__, prediction, "macro")
                file_classifier.write("{0}, {1}, {2}, {3}, {4}\n".format(index, kernel, C, degree, gamma))
                file_results.write("{0},{1},{2},{3},{4}\n".format(index, accuracy, precision, recall, f1))
                index += 1
        file_results.close()
        file_classifier.close()