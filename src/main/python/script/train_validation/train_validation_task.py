# coding: utf-8

import os

import pandas as pd
from sklearn.externals import joblib


class TrainValidationTask:

    def __init__(self, path_save, id_column):
        self.path_save = path_save
        self.id_column = id_column
        self.best_model = None
        self.best_score = -1.0 * float("inf")
        self.best_submission = None
        self.best_prediction = None

        self.file_results = None
        self.file_classifier = None

    def __str__(self):
        s = "Train Validation Task"
        return s

    def define_best_model(self, score, model):
        if score > self.best_score:
            self.best_score = score
            self.best_model = model

    def save_best_model(self):
        print("  Best score: {0}".format(self.best_score))
        joblib.dump(self.best_model,
                    os.path.join(self.path_save, "best_model.pkl"))

    def submission_best_model(self, X):
        self.best_submission = self.best_model.predict(X)

    def prediction_best_model(self, X):
        self.best_prediction = self.best_model.predict(X)

    def save_submission_best_model(self):
        (pd.DataFrame({"Id": self.id_column, "Target": self.best_submission})
         .to_csv(os.path.join(self.path_save, "submission.csv"), index=False))

    def save_prediction_best_model(self, label):
        (pd.DataFrame({"label": label, "prediction": self.best_prediction})
         .to_csv(os.path.join(self.path_save, "prediction.csv"), index=False))


    def open_scores_file(self):
        if not os.path.isdir(self.path_save): os.makedirs(self.path_save)
        path_results = os.path.join(self.path_save, "results.csv")
        if os.path.exists(path_results): os.remove(path_results)
        self.file_results = open(path_results, "a+")
        self.file_results.write("index,accuracy,precision,recall,f1\n")

    def write_scores_file(self, index, accuracy, precision, recall, f1):
        self.file_results.write("{0},{1},{2},{3},{4}\n".format(index, accuracy,
                                                               precision,
                                                               recall, f1))

    def close_scores_file(self):
        self.file_results.close()

    def open_estimator_parameters_file(self):
        if not os.path.isdir(self.path_save): os.makedirs(self.path_save)
        path_classifier = os.path.join(self.path_save, "estimators_parameters.csv")
        if os.path.exists(path_classifier): os.remove(path_classifier)
        self.file_classifier = open(path_classifier, "a+")
        self.file_classifier.write("index;estimators\n")

    def write_estimator_parameters_file(self, index, estimator):
        self.file_classifier.write("{0};{1}\n".format(
            index, str(estimator.get_params())))

    def close_estimator_parameters_file(self):
        self.file_classifier.close()
