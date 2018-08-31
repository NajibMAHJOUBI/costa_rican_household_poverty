# coding: utf-8

import pandas as pd

from classification.nearest_neighbors import KNeighborsClassifierTask
from grid_parameters.grid_nearest_neighbors import get_grid_parameters
from scores import all_scores


class TrainValidationKNN:

    def __init__(self, X_train, X_validation, y_train, y_validation, save_path):
        self.__X_train__ = X_train
        self.__X_validation__ = X_validation
        self.__y_train__ = y_train
        self.__y_validation__ = y_validation
        self.__self_path__ = save_path

    def __str__(self):
        s = "Train Validation Evaluator"
        return s

    def grid_parameters(self, dic_parameters):
        return [(neighbor, weight, algo, leaf, distance)
                for neighbor in dic_parameters["n_neighbors"]
                for weight in dic_parameters["weights"]
                for algo in dic_parameters["algorithm"]
                for leaf in dic_parameters["leaf_size"]
                for distance in dic_parameters["metric"]]

    def run(self):
        dic_results = {"n_neighbors": [], "weights": [], "algorithm": [], "leaf_size": [], "metric": [],
                       "accuracy": {data_set: [] for data_set in ["train", "validation"]},
                       "precision": {data_set: [] for data_set in ["train", "validation"]},
                       "recall": {data_set: [] for data_set in ["train", "validation"]},
                       "f1": {data_set: [] for data_set in ["train", "validation"]},}
        for grid_param in get_grid_parameters():
            for neighbor, weight, algo, leaf, distance in self.grid_parameters(grid_param):
                print("Neighbor: {0}, Weight: {1}, Algorithm: {2}, "
                      "Leaf_size: {3}, Distance: {4}".format(neighbor, weight,
                                                             algo, leaf,
                                                             distance))
                classifier = KNeighborsClassifierTask(n_neighbors=neighbor,
                                                      algorithm=algo,
                                                      weights=weight,
                                                      leaf_size=leaf,
                                                      metric=distance)
                classifier.define_estimator()
                classifier.fit(self.__X_train__, self.__y_train__)
                prediction_train = classifier.predict(self.__X_train__)
                prediction_validation = classifier.predict(self.__X_validation__)

                accuracy_train, precision_train, recall_train, f1_train = all_scores.all_scores(self.__y_train__, prediction_train, "macro")
                accuracy_validation, precision_validation, recall_validation, f1_validation = all_scores.all_scores(self.__y_validation__, prediction_validation, "macro")

                dic_results["n_neighbors"].append(neighbor)
                dic_results["weights"].append(weight)
                dic_results["algorithm"].append(algo)
                dic_results["leaf_size"].append(leaf)
                dic_results["metric"].append(distance)

                dic_results["accuracy"]["train"].append(accuracy_train)
                dic_results["accuracy"]["validation"].append(accuracy_validation)

                dic_results["precision"]["train"].append(precision_train)
                dic_results["precision"]["validation"].append(precision_validation)

                dic_results["recall"]["train"].append(recall_train)
                dic_results["recall"]["validation"].append(recall_validation)

                dic_results["f1"]["train"].append(f1_train)
                dic_results["f1"]["validation"].append(f1_validation)

        self.save_result(dic_results, self.__self_path__)

    def save_result(self, dic_results, path):
        d = {
            "n_neighbors": dic_results["n_neighbors"],
            "weights": dic_results["weights"],
            "algorithm": dic_results["algorithm"],
            "leaf_size": dic_results["leaf_size"],
            "metric": dic_results["metric"],
            "accuracy_train": dic_results["accuracy"]["train"],
            "accuracy_validation": dic_results["accuracy"]["validation"],
            "precision_train": dic_results["precision"]["train"],
            "precision_validation": dic_results["precision"]["validation"],
            "recall_train": dic_results["recall"]["train"],
            "recall_validation": dic_results["recall"]["validation"],
            "f1_train": dic_results["f1"]["train"],
            "f1_validation": dic_results["f1"]["validation"]
        }
        pd.DataFrame(d).to_csv(self.__self_path__, index=False)