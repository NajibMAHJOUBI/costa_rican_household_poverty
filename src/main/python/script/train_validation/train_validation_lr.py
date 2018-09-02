# coding: utf-8

import os
import pandas as pd
from itertools import product

from classification.logistic_regression import LogisticRegressionTask
from grid_parameters.grid_logistic_regression import get_grid_parameters
from scores import all_scores


class TrainValidationLR:

    def __init__(self, X_train, X_validation, y_train, y_validation, save_path):
        self.__X_train__ = X_train
        self.__X_validation__ = X_validation
        self.__y_train__ = y_train
        self.__y_validation__ = y_validation
        self.__save_path__ = save_path

    def __str__(self):
        s = "Train Validation Evaluator"
        return s

        # {"penalty": ["l2"],
        #  "dual": [True, False],
        #  "C": [100.0, 10.0, 1.0, 0.1, 0.01],
        #  "fit_intercept": [True, False]},

    def run(self):
        file_results = open(os.path.join(self.__save_path__, "results.csv"), "a+")
        file_results.write("index,accuracy,precision,recall,f1\n")

        file_classifier = open(os.path.join(self.__save_path__, "classifier.csv"), "a+")
        file_classifier.write("index,penalty,dual,C,fit_intercept\n")

        index = 0
        for params in get_grid_parameters():
            for penalty, dual, C, fit_intercept in product(params["penalty"], params["dual"], params["C"], params["fit_intercept"]):
                print("{0},{1},{2},{3},{4}\n".format(index, penalty, dual, C, fit_intercept))
                classifier = LogisticRegressionTask(penalty=penalty,
                                                    dual=dual,
                                                    C=C,
                                                    fit_intercept=fit_intercept)
                classifier.define_estimator()
                classifier.fit(self.__X_train__, self.__y_train__)
                prediction = classifier.predict(self.__X_validation__)
                accuracy, precision, recall, f1 = all_scores.all_scores(self.__y_validation__, prediction, "macro")
                file_classifier.write("{0},{1},{2},{3},{4}\n".format(index, penalty, dual, C, fit_intercept))
                file_results.write("{0},{1},{2},{3},{4}\n".format(index, accuracy, precision, recall, f1))
                index += 1

        file_results.close()
        file_classifier.close()
