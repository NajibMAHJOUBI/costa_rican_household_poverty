# coding: utf-8

import os
import sys
sys.path.append(os.path.realpath('../..'))

# import libraries
from utils.load_data_task import LoadDataTask
from utils.define_label_features import DefineLabelFeaturesTask
from over_sampling.over_sampling_task import OverSamplingTask
from split_task.train_test_split import TrainTestSplit
from features_selector.pearson_selector import PearsonSelectorTask
from fill_na.fill_na import FillNaValuesTask
from standard_scaler.standard_scaler import StandardScalerTask
from train_validation.train_validation_nearest_neighbors import TrainValidationKNearestNeighbors
from train_validation.train_validation_logistic_regression import TrainValidationLogisticRegression
from train_validation.train_validation_random_forest import TrainValidationRandomForest
from train_validation.train_validation_decision_tree_classifier import TrainValidationDecisionTreeClassifier
from train_validation.train_validation_extra_trees import TrainValidationExtraTreesClassifier
from train_validation.train_validation_one_vs_rest import TrainValidationOneVsRest
from train_validation.train_validation_xgboost import TrainValidationXGBoost


# continuous features
print("Load datasets")
continuous_features = open("../../../../resources/continuousFeatures").read().split(",")
print("  Initial number of continuous features: {0}".format(len(continuous_features)))

# Load datasets
path_data = "../../../../../../data"
# --> train
load_data_task = LoadDataTask(os.path.join(path_data, "train/train.csv"))
train = load_data_task.load_data()[["Id", "Target"] + continuous_features]
# --> test
load_data_task = LoadDataTask(os.path.join(path_data, "test/test.csv"))
test = load_data_task.load_data()[["Id"] + continuous_features]

print("  Train shape: {0}".format(train.shape))
print("  Test shape: {0}".format(test.shape))

# Fill Nan
print("Fill Nan tasks")
fill_na_values = FillNaValuesTask(train, test, continuous_features, "mean")
fill_na_values.fill_by_values()
yes_no_features = open("../../../../resources/yesNoFeaturesNames").read().split(",")
fill_na_values.fill_yes_no(yes_no_features)
train = fill_na_values.get_train()
test = fill_na_values.get_test()

# Pearson continuous selector
print("Features selection")
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

# define features and label
print("Define label-features data")
define_label_features = DefineLabelFeaturesTask("Id", "Target", continuous_features)
train_id = define_label_features.get_id(train)
train_label  = define_label_features.get_label(train)
train_features = define_label_features.get_features(train)

test_id = define_label_features.get_id(test)
test_features = define_label_features.get_features(test)

# Standard scaler
scaler = StandardScalerTask(train_features)
scaler.define_estimator()
scaler.fit()
train_features = scaler.transform(train_features)
X_test = scaler.transform(test_features)

# train-validation split
print("Train-Validation split")
train_test_split = TrainTestSplit(train_features, train_label, test_size=0.3, stratify=train_label)
X_train, X_validation, y_train, y_validation = train_test_split.split()

# smote sampling
print("Over-sampling: SMOTE")
over_sampling = OverSamplingTask(X_train, y_train)
X_resampled, y_resampled = over_sampling.smote()

# Train Validation -
print("Train Validation process")
score = "f1"
save_path = os.path.join(os.path.realpath("../../../../../.."), "submission/sklearn/train_validation/continuous")
# base_estimators = ["nearest_neighbors", "logistic_regression", "decision_tree", "random_forest", "extra_trees"]
base_estimators = ["decision_tree", "random_forest", "extra_trees"]
for classifier in base_estimators:
    print("Classifier: {0}".format(classifier))
    train_validation = None
    if classifier == "nearest_neighbors":
        train_validation = TrainValidationKNearestNeighbors(X_resampled, X_validation, y_resampled, y_validation,
                                                            "euclidean", save_path)
    elif classifier == "logistic_regression":
        train_validation = TrainValidationLogisticRegression(X_resampled, X_validation, y_resampled, y_validation,
                                                             save_path)
    elif classifier == "random_forest":
        train_validation = TrainValidationRandomForest(X_resampled, X_validation, y_resampled, y_validation, X_test,
                                                       test_id, score, save_path)
    elif classifier == "decision_tree":
        train_validation = TrainValidationDecisionTreeClassifier(X_resampled, X_validation, y_resampled, y_validation,
                                                                 X_test, test_id, score, save_path)
    elif classifier == "extra_trees":
        train_validation = TrainValidationExtraTreesClassifier(X_resampled, X_validation, y_resampled, y_validation,
                                                               X_test, test_id, score, save_path)
    train_validation.run()

classifier = "one_vs_rest"
print("Classifier: {0}".format(classifier))
for base_estimator in ["decision_tree", "random_forest"]:
    print("Base estimator: {0}".format(base_estimator))
    train_validation = TrainValidationOneVsRest(X_resampled, X_validation, y_resampled, y_validation, X_test, test_id,
                                                base_estimator, score, save_path, metric="euclidean")
    train_validation.run()

# classifier = "ada_boosting"
# print("Classifier: {0}".format(classifier))
# for type_classifier in ["random_forest"]:
#     print("Base estimator: {0}".format(type_classifier))
#     save_path = get_path_save(classifier, pearson_option, estimator=type_classifier)
#     train_validation = TrainValidationAB(X_resampled, X_validation, y_resampled, y_validation, type_classifier, save_path, metric="euclidean")
#     train_validation.run()

classifier = "xgboost"
print("Classifier: {0}".format(classifier))
train_validation = TrainValidationXGBoost(X_resampled, X_validation, y_resampled, y_validation, X_test, test_id, score,
                                          save_path)
train_validation.run()
