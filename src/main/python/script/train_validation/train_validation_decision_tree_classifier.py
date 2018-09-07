# coding: utf-8

import os
from itertools import product

from classification.decision_tree_classifier import DecisionTreeClassifierTask
from grid_parameters.grid_decision_tree import get_grid_parameters
from scores import all_scores

from train_validation_task import TrainValidationTask


class TrainValidationDecisionTreeClassifier(TrainValidationTask):

    def __init__(self, X_train, X_validation, y_train, y_validation, X_test, id_test, score, save_path):
        TrainValidationTask.__init__(self, os.path.join(save_path, "decision_tree"), id_test)
        self.__X_train__ = X_train
        self.__X_validation__ = X_validation
        self.__X_test__ = X_test
        self.__y_train__ = y_train
        self.__y_validation__ = y_validation
        self.__score__ = score
        self.__save_path__ = save_path

    def __str__(self):
        s = "Train Validation - Decision Tree Classifier "
        return s

    def run(self):
        self.open_scores_file()
        self.open_estimator_parameters_file()

        index = 0
        for params in get_grid_parameters():
            for criteria, depth, split, leaf in product(params["criterion"],
                                                        params["max_depth"],
                                                        params["min_samples_split"],
                                                        params["min_samples_leaf"]):
                # print("{0},{1},{2},{3},{4}\n".format(index, criteria, depth, split, leaf))
                classifier = DecisionTreeClassifierTask(criterion=criteria,
                                                        max_depth=depth,
                                                        min_samples_split=split,
                                                        min_samples_leaf=leaf)
                classifier.define_estimator()
                classifier.fit(self.__X_train__, self.__y_train__)
                prediction = classifier.predict(self.__X_validation__)
                accuracy, precision, recall, f1 = all_scores.all_scores(self.__y_validation__, prediction, "macro")

                self.write_estimator_parameters_file(index, classifier.get_estimator())
                self.write_scores_file(index, accuracy, precision, recall, f1)

                if self.__score__ == "f1":
                    self.define_best_model(f1, classifier.get_model())
                elif self.__score__ == "accuracy":
                    self.define_best_model(accuracy, classifier.get_model())
                elif self.__score__ == "precision":
                    self.define_best_model(precision, classifier.get_model())
                elif self.__score__ == "recall":
                    self.define_best_model(recall, classifier.get_model())

                index += 1

        self.close_scores_file()
        self.close_estimator_parameters_file()
        self.save_best_model()
        self.prediction_best_model(self.__X_validation__)
        self.submission_best_model(self.__X_test__)
        self.save_prediction_best_model(self.__y_validation__)
        self.save_submission_best_model()
