# coding: utf-8

from itertools import product

from classification.decision_tree import DecisionTreeTask
from classification.logistic_regression import LogisticRegressionTask
from classification.nearest_neighbors import KNeighborsClassifierTask
from classification.random_forest import RandomForestTask
from classification.svc import SVCTask

import grid_decision_tree
import grid_logistic_regression
import grid_nearest_neighbors
import grid_random_forest
import grid_svc


def get_grid_parameters(base_estimator, metric=None):
    if base_estimator == "random_forest":
        return [{"estimator": get_random_forest()}]
    elif base_estimator == "decision_tree":
        return [{"estimator": get_decision_tree()}]
    elif base_estimator == "logistic_regression":
        return [{"estimator": get_logistic_regression()}]
    elif base_estimator == "nearest_neighbors":
        return [{"estimator": get_nearest_neighbors(metric)}]
    elif base_estimator == "svc":
        return [{"estimator": get_svc()}]


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


def get_decision_tree():
    grid_parameters = grid_decision_tree.get_grid_parameters()[0]
    cartesian_product = product(grid_parameters["criterion"],
                                grid_parameters["max_depth"],
                                grid_parameters["min_samples_split"],
                                grid_parameters["min_samples_leaf"])
    list_estimators = []
    for criteria,depth,split,leaf in cartesian_product:
        classifier = DecisionTreeTask(criterion=criteria,
                                      max_depth=depth,
                                      min_samples_split=split,
                                      min_samples_leaf=leaf)
        classifier.define_estimator()
        list_estimators.append(classifier.get_estimator())
    return list_estimators


def get_logistic_regression():
    dic_grid_parameters = grid_logistic_regression.get_grid_parameters()
    list_estimators = []
    for dic_params in dic_grid_parameters:
        cartesian_product = product(dic_params["penalty"],
                                    dic_params["dual"],
                                    dic_params["C"],
                                    dic_params["fit_intercept"])
        for penalty, dual, C, fit_intercept in cartesian_product:
            classifier = LogisticRegressionTask(penalty=penalty,
                                                dual=dual,
                                                C=C,
                                                fit_intercept=fit_intercept)
            classifier.define_estimator()
            list_estimators.append(classifier.get_estimator())
    return list_estimators


def get_nearest_neighbors(metric):
    dic_grid_parameters = grid_nearest_neighbors.get_grid_parameters(metric)
    list_estimators = []
    for dic_params in dic_grid_parameters:
        cartesian_product = product(dic_params["n_neighbors"],
                                    dic_params["weights"],
                                    dic_params["algorithm"],
                                    dic_params["leaf_size"],
                                    dic_params["metric"])
        for n_neighbors, weights, algorithm, leaf_size, metric in cartesian_product:
            classifier = KNeighborsClassifierTask(n_neighbors=n_neighbors,
                                                  algorithm=algorithm,
                                                  weights=weights,
                                                  leaf_size=leaf_size,
                                                  metric=metric)
            classifier.define_estimator()
            list_estimators.append(classifier.get_estimator())
    return list_estimators


def get_svc():
    dic_grid_parameters = grid_svc.get_grid_parameters()
    list_estimators = []
    for dic_params in dic_grid_parameters:
        cartesian_product = product(dic_params["kernel"],
                                    dic_params["degree"],
                                    dic_params["gamma"],
                                    dic_params["C"])
        for kernel, degree, gamma, C in cartesian_product:
            classifier = SVCTask(kernel=kernel,
                                 C=C,
                                 degree=degree,
                                 gamma=gamma)
            classifier.define_estimator()
            list_estimators.append(classifier.get_estimator())
    return list_estimators
