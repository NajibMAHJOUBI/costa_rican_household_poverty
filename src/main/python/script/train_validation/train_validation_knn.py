# coding: utf-8

import os
import pandas as pd
from itertools import product

from classification.nearest_neighbors import KNeighborsClassifierTask
from grid_parameters.grid_nearest_neighbors import get_grid_parameters
from scores import all_scores


class TrainValidationKNN:

    def __init__(self, X_train, X_validation, y_train, y_validation, metric, save_path):
        self.__X_train__ = X_train
        self.__X_validation__ = X_validation
        self.__y_train__ = y_train
        self.__y_validation__ = y_validation
        self.__metric__ = metric
        self.__save_path__ = save_path

    def __str__(self):
        s = "Train Validation Evaluator - k Nearest Neighbors"
        return s

    def run(self):
        file_results = open(os.path.join(self.__save_path__, "results.csv"), "a+")
        file_results.write("index,accuracy,precision,recall,f1\n")

        file_classifier = open(os.path.join(self.__save_path__, "classifier.csv"), "a+")
        file_classifier.write("index,n_neighbors,weights,algorithm,leaf_size,metric\n")

        index = 0
        for params in get_grid_parameters(self.__metric__):
            for neighbor, weight, algo, leaf, distance in product(params["n_neighbors"],
                                                                  params["weights"],
                                                                  params["algorithm"],
                                                                  params["leaf_size"],
                                                                  params["metric"]):
                print("{0}, {1}, {2}, {3}, {4}, {5}".format(index, neighbor, weight, algo, leaf, distance))
                classifier = KNeighborsClassifierTask(n_neighbors=neighbor,
                                                      algorithm=algo,
                                                      weights=weight,
                                                      leaf_size=leaf,
                                                      metric=distance)
                classifier.define_estimator()
                classifier.fit(self.__X_train__, self.__y_train__)
                prediction = classifier.predict(self.__X_validation__)
                accuracy, precision, recall, f1 = all_scores.all_scores(self.__y_validation__, prediction, "macro")

                file_classifier.write("{0}, {1}, {2}, {3}, {4}, {5}".format(index, neighbor, weight, algo, leaf, distance))
                file_results.write("{0},{1},{2},{3},{4}\n".format(index, accuracy, precision, recall, f1))
                index += 1

        file_results.close()
        file_classifier.close()
