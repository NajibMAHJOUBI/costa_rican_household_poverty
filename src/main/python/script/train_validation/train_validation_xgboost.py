# coding: utf-8

import os
from itertools import product

from classification.xgboost_classifier import XGBoostClassifierTask
from grid_parameters.grid_xgboost import get_grid_parameters
from scores import all_scores


class TrainValidationXGBoost:

    def __init__(self, X_train, X_validation, y_train, y_validation, save_path):
        self.__X_train__ = X_train
        self.__X_validation__ = X_validation
        self.__y_train__ = y_train
        self.__y_validation__ = y_validation
        self.__save_path__ = save_path

    def __str__(self):
        s = "Train Validation Evaluator"
        return s

    def run(self):
        file_results = open(os.path.join(self.__save_path__, "results.csv"), "a+")
        file_results.write("index,accuracy,precision,recall,f1\n")

        file_classifier = open(os.path.join(self.__save_path__, "classifier.csv"), "a+")
        file_classifier.write("index,max_depth,n_estimators,learning_rate\n")

        # 'max_depth': [2, 4, 6], 'n_estimators'
        index = 0
        for params in get_grid_parameters():
            for max_depth, n_estimators, learning_rate in product(params["max_depth"], params["n_estimators"], params["learning_rate"]):
                # print("{0},{1}\n".format(index, estimator))
                classifier = XGBoostClassifierTask(max_depth=max_depth,
                                                   n_estimators=n_estimators,
                                                   learning_rate=learning_rate)
                classifier.define_estimator()
                classifier.fit(self.__X_train__, self.__y_train__)
                prediction = classifier.predict(self.__X_validation__)
                accuracy, precision, recall, f1 = all_scores.all_scores(self.__y_validation__, prediction, "macro")
                file_classifier.write("{0},{1},{2},{3}\n".format(index, max_depth, n_estimators, learning_rate))
                file_results.write("{0},{1},{2},{3},{4}\n".format(index, accuracy, precision, recall, f1))
                index += 1

        file_results.close()
        file_classifier.close()
