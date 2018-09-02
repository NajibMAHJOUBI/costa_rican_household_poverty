# coding: utf-8

import os
import pandas as pd
from itertools import product

from classification.one_vs_rest import OneVsRestTask
from grid_parameters.grid_one_vs_rest import get_grid_parameters
from scores import all_scores


class TrainValidationOVR:

    def __init__(self, X_train, X_validation, y_train, y_validation, type_classifier, save_path):
        self.__X_train__ = X_train
        self.__X_validation__ = X_validation
        self.__y_train__ = y_train
        self.__y_validation__ = y_validation
        self.__type_classifier__ = type_classifier
        self.__save_path__ = save_path

    def __str__(self):
        s = "Train Validation Evaluator"
        return s

    def run(self):
        file_results = open(os.path.join(self.__save_path__, "results.csv"), "a+")
        file_results.write("index,accuracy,precision,recall,f1\n")

        file_classifier = open(os.path.join(self.__save_path__, "classifier.csv"), "a+")
        file_classifier.write("index,n_estimators,criterion,max_depth,min_samples_split,min_samples_leaf\n")

        index = 0
        for params in get_grid_parameters(self.__type_classifier__):
            for estimator in params["estimator"]:
                print("{0},{1},{2},{3},{4},{5}\n".format(index, estimator.n_estimators, estimator.criterion,
                                                         estimator.max_depth, estimator.min_samples_split,
                                                         estimator.min_samples_leaf))
                classifier = OneVsRestTask()
                classifier.set_classifier(estimator)
                classifier.define_estimator()
                classifier.fit(self.__X_train__, self.__y_train__)
                prediction = classifier.predict(self.__X_validation__)
                accuracy, precision, recall, f1 = all_scores.all_scores(self.__y_validation__, prediction, "macro")
                file_classifier.write("{0},{1},{2},{3},{4},{5}\n".format(index, estimator.n_estimators, estimator.criterion,
                                                         estimator.max_depth, estimator.min_samples_split,
                                                         estimator.min_samples_leaf))
                file_results.write("{0},{1},{2},{3},{4}\n".format(index, accuracy, precision, recall, f1))
                index += 1

        file_results.close()
        file_classifier.close()
