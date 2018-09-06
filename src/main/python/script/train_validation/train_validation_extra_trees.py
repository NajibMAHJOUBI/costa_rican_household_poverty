# coding: utf-8

import os
from itertools import product

from classification.extra_trees_classifier import ExtraTreesClassifierTask
from grid_parameters.grid_extra_trees import get_grid_parameters
from scores import all_scores


class TrainValidationExtraTreesClassifier:

    def __init__(self, X_train, X_validation, y_train, y_validation, save_path):
        self.__X_train__ = X_train
        self.__X_validation__ = X_validation
        self.__y_train__ = y_train
        self.__y_validation__ = y_validation
        self.__save_path__ = save_path

    def __str__(self):
        s = "Train Validation Evaluator - Extra Trees Classifier"
        return s

    def run(self):
        path_results = os.path.join(self.__save_path__, "results.csv")
        if os.path.exists(path_results): os.remove(path_results)
        file_results = open(path_results, "a+")
        file_results.write("index,accuracy,precision,recall,f1\n")

        path_classifier = os.path.join(self.__save_path__, "classifier.csv")
        if os.path.exists(path_classifier): os.remove(path_classifier)
        file_classifier = open(path_classifier, "a+")
        file_classifier.write("index; estimator\n")

        index = 0
        for params in get_grid_parameters():
            for estimator, criteria, depth, split, leaf in product(params["n_estimators"],
                                                                   params["criterion"],
                                                                   params["max_depth"],
                                                                   params["min_samples_split"],
                                                                   params["min_samples_leaf"]):
                # print("{0},{1},{2},{3},{4},{5}\n".format(index, estimator, criteria, depth, split, leaf))
                classifier = ExtraTreesClassifierTask(n_estimators=estimator,
                                                      criterion=criteria,
                                                      max_depth=depth,
                                                      min_samples_split=split,
                                                      min_samples_leaf=leaf)
                classifier.define_estimator()
                classifier.fit(self.__X_train__, self.__y_train__)
                prediction = classifier.predict(self.__X_validation__)
                accuracy, precision, recall, f1 = all_scores.all_scores(self.__y_validation__, prediction, "macro")
                file_classifier.write("{0}; {1}\n".format(index, classifier.get_estimator().get_params()))
                file_results.write("{0},{1},{2},{3},{4}\n".format(index, accuracy, precision, recall, f1))
                index += 1

        file_results.close()
        file_classifier.close()
