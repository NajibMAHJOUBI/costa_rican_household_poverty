# coding: utf-8

from classification.random_forest import RandomForestTask
from itertools import product

import grid_random_forest


def get_grid_parameters(estimator):
    if estimator == "random_forest":
        return [{"estimator": get_random_forest()}]


def get_random_forest():
    grid_parameters = grid_random_forest.get_grid_parameters()[0]
    cartesian_product = product(grid_parameters["n_estimators"],
                                grid_parameters["criterion"],
                                grid_parameters["max_depth"],
                                grid_parameters["min_samples_split"],
                                grid_parameters["min_samples_leaf"])
    list_estimators = []
    for n_estimator, criteria, depth, split, leaf in cartesian_product:
        classifier = RandomForestTask(n_estimators=n_estimator, criterion=criteria,
                                      max_depth=depth, min_samples_split=split, min_samples_leaf=leaf)
        classifier.define_estimator()
        list_estimators.append(classifier.get_estimator())
    return list_estimators
