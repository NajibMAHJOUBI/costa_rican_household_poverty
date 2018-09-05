# coding: utf-8


import grid_one_vs_rest


def get_grid_parameters(base_estimator, metric=None):
    n_estimators = [50, 100, 150]
    if base_estimator == "random_forest":
        return [{"base_estimator": grid_one_vs_rest.get_random_forest(), "n_estimators": n_estimators}]
    elif base_estimator == "decision_tree":
        return [{"base_estimator": grid_one_vs_rest.get_decision_tree(), "n_estimators": n_estimators}]
    elif base_estimator == "logistic_regression":
        return [{"base_estimator": grid_one_vs_rest.get_logistic_regression(), "n_estimators": n_estimators}]
    elif base_estimator == "nearest_neighbors":
        return [{"base_estimator": grid_one_vs_rest.get_nearest_neighbors(metric), "n_estimators": n_estimators}]
    elif base_estimator == "svc":
        return [{"base_estimator": grid_one_vs_rest.get_svc(), "n_estimators": n_estimators}]
