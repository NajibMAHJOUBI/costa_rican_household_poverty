# coding: utf-8

import os
import pandas as pd
from itertools import product

from classification.random_forest import RandomForestTask
from grid_parameters.grid_random_forest import get_grid_parameters
from scores import all_scores


class TrainValidationRF:

    def __init__(self, X_train, X_validation, y_train, y_validation, metric, save_path):
        self.__X_train__ = X_train
        self.__X_validation__ = X_validation
        self.__y_train__ = y_train
        self.__y_validation__ = y_validation
        self.__metric__ = metric
        self.__save_path__ = save_path

    def __str__(self):
        s = "Train Validation Evaluator - Random Forest"
        return s

    def run(self):
        file_results = open(os.path.join(self.__save_path__, "results.csv"), "a+")
        file_results.write("index,accuracy,precision,recall,f1\n")

        file_classifier = open(os.path.join(self.__save_path__, "classifier.csv"), "a+")
        file_classifier.write("index,n_estimators,criterion,max_depth,min_samples_split,min_samples_leaf\n")

        index = 0
        for params in get_grid_parameters(self.__metric__):
            for estimator, criteria, depth, split, leaf in product(params["n_estimators"],
                                                                   params["criterion"],
                                                                   params["max_depth"],
                                                                   params["min_samples_split"],
                                                                   params["min_samples_leaf"]):
                print("{0},{1},{2},{3},{4},{5}\n".format(index, estimator, criteria, depth, split, leaf))
                classifier = RandomForestTask(n_estimators=estimator,
                                              criterion=criteria,
                                              max_depth=depth,
                                              min_samples_split=split,
                                              min_samples_leaf=leaf)
                classifier.define_estimator()
                classifier.fit(self.__X_train__, self.__y_train__)
                prediction = classifier.predict(self.__X_validation__)
                accuracy, precision, recall, f1 = all_scores.all_scores(self.__y_validation__, prediction, "macro")
                file_classifier.write("{0},{1},{2},{3},{4},{5}\n".format(index, estimator, criteria, depth, split, leaf))
                file_results.write("{0},{1},{2},{3},{4}\n".format(index, accuracy, precision, recall, f1))
                index += 1

        file_results.close()
        file_classifier.close()
