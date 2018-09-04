# coding: utf-8

import os
import sys
sys.path.append(os.path.realpath('../..'))
import pandas as pd

# import libraries
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
from classification.ada_boost_classifier import AdaBoostClassifierTask
from classification.xgboost_classifier import XGBoostClassifierTask
from split_task.train_test_split import TrainTestSplit
from over_sampling.over_sampling_task import OverSamplingTask
from scores import all_scores
from features_selector.pearson_selector import PearsonSelectorTask
from fill_na.fill_na import FillNaValuesTask
from standard_scaler.standard_scaler import StandardScalerTask

# continuous features
continuous_features = open("../../../../resources/continuousFeatures").read().split(",")
print("Initial number of continuous features: {0}".format(len(continuous_features)))

# Load datasets
path_data = "../../../../../../data"
# --> train
load_data_task = LoadDataTask(os.path.join(path_data, "train/train.csv"))
train = load_data_task.load_data()[["Id", "Target"] + continuous_features]
# --> test
load_data_task = LoadDataTask(os.path.join(path_data, "test/test.csv"))
test = load_data_task.load_data()[["Id"] + continuous_features]

print("Train shape: {0}".format(train.shape))
print("Test shape: {0}".format(test.shape))

# Fill Nan
fill_na_values = FillNaValuesTask(train, test, continuous_features, "mean")
fill_na_values.fill_by_values()
yes_no_features = open("../../../../resources/yesNoFeaturesNames").read().split(",")
fill_na_values.fill_yes_no(yes_no_features)
train = fill_na_values.get_train()
test = fill_na_values.get_test()

# Pearson continuous selector
pearson_option = True
if pearson_option:
    # Pearson selection
    pearson_selector = PearsonSelectorTask(train, "Id", "Target", continuous_features, 0.95)
    pearson_selector.define_restrained_features()
    train = pearson_selector.get_restrained_features(train, "train")
    test = pearson_selector.get_restrained_features(test, "test")
    continuous_features = pearson_selector.get_restrained_columns()
    print("Train shape: {0}".format(train.shape))
    print("Test shape: {0}".format(test.shape))

# Define label features for train and test datasets
define_label_features = DefineLabelFeaturesTask("Id", "Target", continuous_features)
train_id = define_label_features.get_id(train)
train_label = define_label_features.get_label(train)
train_features = define_label_features.get_features(train)

test_id = define_label_features.get_id(test)
test_features = define_label_features.get_features(test)

# Standard scaler
scaler = StandardScalerTask(train_features)
scaler.define_estimator()
scaler.fit()
train_features = scaler.transform(train_features)

# Train-Validation split
train_test_split = TrainTestSplit(train_features, train_label,
                                  test_size=0.3, stratify=train_label)
X_train, X_validation, y_train, y_validation = train_test_split.split()

# smote sampling
over_sampling = OverSamplingTask(X_train, y_train)
X_resampled, y_resampled = over_sampling.smote()



# Loop over classifier list
classifier_list = ["decision_tree", "random_forest", "logistic_regression",
                   "nearest_neighbors", "gaussian_nb", "mlp_classifier",
                   "one_vs_rest", "quadratic_discriminant", "svc", "ada_boost",
                   "xgboost"]

dic_results = {"classifier": [], "accuracy": [], "precision": [], "recall": [], "f1": []}
for classifier in classifier_list:
    algorithm = None
    print("Classifier: {0}".format(classifier))
    if classifier == "decision_tree":
        algorithm = DecisionTreeTask()
    elif classifier == "random_forest":
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
        algorithm = OneVsRestTask()
        algorithm.base_estimator("random_forest")
    elif classifier == "quadratic_discriminant":
        algorithm = QuadraticDiscriminantAnalysisTask()
    elif classifier == "svc":
        algorithm = SVCTask()
    elif classifier == "ada_boost":
        algorithm = AdaBoostClassifierTask()
    elif classifier == "xgboost":
        algorithm = XGBoostClassifierTask()

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
if pearson_option:
    path = os.path.realpath('../../../../../../submission/sklearn/classifier/continuous/scaled/pearson_features.csv')
else:
    path = os.path.realpath('../../../../../../submission/sklearn/classifier/continuous/scaled/all_features.csv')

pd.DataFrame(dic_results).to_csv(path, index=False)
