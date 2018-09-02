# coding: utf-8

import os
import sys
sys.path.append(os.path.realpath('..'))
import pandas as pd

from utils.load_data_task import LoadDataTask
from utils.define_label_features import DefineLabelFeaturesTask
from classification.random_forest import RandomForestTask
from classification.decision_tree import DecisionTreeTask
from classification.nearest_neighbors import KNeighborsClassifierTask
from classification.logistic_regression import LogisticRegressionTask
from classification.gaussian_nb import GaussianNBTask
from classification.mlp_classifier import MLPClassifierTask
from classification.one_vs_rest import OneVsRestTask
from classification.quadratic_discriminant_analysis import QuadraticDiscriminantAnalysisTask
from classification.svc import SVCTask
from split_task.train_test_split import TrainTestSplit
from utils.build_submission import build_submission
from over_sampling.over_sampling_task import OverSamplingTask
from scores import all_scores
from features_selector.chi2_selector import Chi2SelectorTask


# Load datasets
path_data = "../../../../../data"
# --> train
load_data_task = LoadDataTask(os.path.join(path_data, "train/train.csv"))
train = load_data_task.load_data()
# --> test
load_data_task = LoadDataTask(os.path.join(path_data, "test/test.csv"))
test = load_data_task.load_data()

print("Train shape: {0}".format(train.shape))
print("Test shape: {0}".format(test.shape))

chi2_option = True
if chi2_option:
    # Chi-square selection
    categorical_features = open("../../../resources/categoricalFeatures").read().split(",")
    chi2_selector = Chi2SelectorTask(train, "Id", "Target", categorical_features, 0.05)
    chi2_selector.define_restrained_features()
    train = chi2_selector.get_restrained_features(train, "train")
    test = chi2_selector.get_restrained_features(test, "test")
    categorical_features = chi2_selector.get_restrained_columns()
    print("Train shape: {0}".format(train.shape))
    print("Test shape: {0}".format(test.shape))

# Define label features for train and test datasets
define_label_features = DefineLabelFeaturesTask("Id", "Target", categorical_features)
train_id = define_label_features.get_id(train)
train_label = define_label_features.get_label(train)
train_features = define_label_features.get_features(train)

test_id = define_label_features.get_id(test)
test_features = define_label_features.get_features(test)

# Train-Validation split
train_test_split = TrainTestSplit(train_features, train_label, test_size=0.3, stratify=train_label)
X_train, X_validation, y_train, y_validation = train_test_split.split()

# smote sampling
over_sampling = OverSamplingTask(X_train, y_train)
X_resampled, y_resampled = over_sampling.smote()

# Loop over classifier list
classifier_list = ["decision_tree", "random_forest_all_features", "logistic_regression", "nearest_neighbors", "gaussian_nb",
                   "mlp_classifier", "one_vs_rest", "quadratic_discriminant", "svc"]
dic_results = {"classifier": [], "accuracy": [], "precision": [], "recall": [], "f1": []}
for classifier in classifier_list:
    algorithm = None
    print("Classifier: {0}".format(classifier))
    if classifier == "decision_tree":
        algorithm = DecisionTreeTask()
    elif classifier == "random_forest_all_features":
        algorithm = RandomForestTask()
    elif classifier == "logistic_regression":
        algorithm = LogisticRegressionTask()
    elif classifier == "nearest_neighbors":
        algorithm = KNeighborsClassifierTask(metric="jaccard")
    elif classifier == "gaussian_nb":
        algorithm = GaussianNBTask()
    elif classifier == "mlp_classifier":
        algorithm = MLPClassifierTask()
    elif classifier == "one_vs_rest":
        algorithm = OneVsRestTask("random_forest_all_features")
    elif classifier == "quadratic_discriminant":
        algorithm = QuadraticDiscriminantAnalysisTask()
    elif classifier == "svc":
        algorithm = SVCTask()

    algorithm.define_estimator()
    algorithm.fit(X_resampled, y_resampled)
    prediction_validation = algorithm.predict(X_validation)
    accuracy, precision, recall, f1 = all_scores.all_scores(y_validation, prediction_validation, "macro")

    dic_results["classifier"].append(classifier)
    dic_results["accuracy"].append(accuracy)
    dic_results["precision"].append(precision)
    dic_results["recall"].append(recall)
    dic_results["f1"].append(f1)

path = None
if chi2_option:
    path = os.path.realpath('../../../../../submission/sklearn/classifier/chi2_features.csv')
else:
    path = os.path.realpath('../../../../../submission/sklearn/classifier/all_features.csv')

pd.DataFrame(dic_results).to_csv(path, index=False)

