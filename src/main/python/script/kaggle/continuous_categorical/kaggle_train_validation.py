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
from features_selector.chi2_selector import Chi2SelectorTask
from fill_na.fill_na import FillNaValuesTask
from train_validation.train_validation_bagging import TrainValidationBagging

# def utils function
def get_path_save(classifier, estimator=None):
    path_classifier = None
    if estimator is not None:
        path_classifier = "{0}/{1}".format(classifier, estimator)
    else:
        path_classifier = classifier
    path = os.path.join(os.path.realpath("../../../../../.."),
                        "submission/sklearn/train_validation/continuous_categorical")

    new_path = os.path.join(path, "{0}".format(path_classifier))
    if not os.path.isdir(new_path):
        os.makedirs(new_path)
    return new_path


# continuous features
print("Continuous features")
continuous_features = open(
    "../../../../resources/continuousFeatures").read().split(",")
print("  Initial number of continuous features: {0}".format(
    len(continuous_features)))

# categorical features
print("Categorical features")
categorical_features = open(
    "../../../../resources/categoricalFeatures").read().split(",")
print("  Initial number of categorical features: {0}".format(
    len(categorical_features)))

# Load datasets
print("Load Train and test datastets")
path_data = "../../../../../../data"
# --> train
load_data_task = LoadDataTask(os.path.join(path_data, "train/train.csv"))
train = load_data_task.load_data()
train_continuous = train[["Id", "Target"] + continuous_features]
train_categorical = train[["Id", "Target"] + categorical_features]
# --> test
load_data_task = LoadDataTask(os.path.join(path_data, "test/test.csv"))
test = load_data_task.load_data()
test_continuous = test[["Id"] + continuous_features]
test_categorical = test[["Id"] + categorical_features]

print(
"  Train Shape: Complete: {0} - Continuous: {1} - Categorical: {2}".format(
    train.shape, train_continuous.shape, train_categorical.shape))
print("  Test Shape: Complete: {0} - Continuous: {1} - Categorical: {2}"
      .format(test.shape, test_continuous.shape, test_categorical.shape))

print("Columns with null values: ")
null_columns = train_continuous.columns[train_continuous.isnull().any()]
print("  Train continuous: {0}".format(null_columns))

null_columns = train_categorical.columns[train_categorical.isnull().any()]
print("  Train categorical: {0}".format(null_columns))

null_columns = test_continuous.columns[test_continuous.isnull().any()]
print("  Test continuous: {0}".format(null_columns))

null_columns = test_categorical.columns[test_categorical.isnull().any()]
print("  Test categorical: {0}".format(null_columns))

# Fill Nan
print("Fill Nan tasks")
fill_na_values = FillNaValuesTask(train_continuous, test_continuous,
                                  continuous_features, "mean")
fill_na_values.fill_by_values()
yes_no_features = open(
    "../../../../resources/yesNoFeaturesNames").read().split(",")
fill_na_values.fill_yes_no(yes_no_features)
train_continuous = fill_na_values.get_train()
test_continuous = fill_na_values.get_test()

print("  Train continuous: {0}"
      .format(train_continuous.columns[train_continuous.isnull().any()]))
print("  Test continuous: {0}"
      .format(test_continuous.columns[test_continuous.isnull().any()]))
#
# Pearson continuous selector
pearson_option = True
if pearson_option:
    print("\n\nFeatures selection - Continuous ")
    # Pearson selection
    pearson_selector = PearsonSelectorTask(train_continuous, "Id", "Target", continuous_features, 0.95)
    pearson_selector.define_restrained_features()
    # print("   Train shape: {0}".format(train_continuous.shape))
    # print("   Test shape: {0}".format(test_continuous.shape))
    train_continuous = pearson_selector.get_restrained_features(train_continuous, "train")
    test_continuous = pearson_selector.get_restrained_features(test_continuous, "test")
    # print("   Train shape: {0}".format(train_continuous.shape))
    # print("   Test shape: {0}".format(test_continuous.shape))
    continuous_features = pearson_selector.get_restrained_columns()


# Features selection - categorical
chi2_option = True
if chi2_option:
    print("\n\nFeatures selection - Categorical")
    # Chi-square selection
    chi2_selector = Chi2SelectorTask(train_categorical, "Id", "Target", categorical_features, 0.05)
    chi2_selector.define_restrained_features()
    # print("   Train shape: {0}".format(train_categorical.shape))
    # print("   Test shape: {0}".format(test_categorical.shape))
    train_categorical = chi2_selector.get_restrained_features(train_categorical, "train")
    test_categorical = chi2_selector.get_restrained_features(test_categorical, "test")
    # print("   Train shape: {0}".format(train_categorical.shape))
    # print("   Test shape: {0}".format(test_categorical.shape))
    categorical_features = chi2_selector.get_restrained_columns()

# Join data sets continuous and categorical
train = train_categorical.join(train_continuous.drop(["Target"], axis=1).set_index("Id"), on="Id")
test = test_categorical.join(test_continuous.set_index("Id"), on="Id")

print(train.shape)
print(test.shape)

# define features and label
print("Define label-features data")
define_label_features = DefineLabelFeaturesTask("Id", "Target", continuous_features+categorical_features)
train_id = define_label_features.get_id(train)
train_label  = define_label_features.get_label(train)
train_features = define_label_features.get_features(train)

test_id = define_label_features.get_id(test)
test_features = define_label_features.get_features(test)
#
# # Standard scaler
# scaler = StandardScalerTask(train_features)
# scaler.define_estimator()
# scaler.fit()
# train_features = scaler.transform(train_features)
#
# train-validation split
print("Train-Validation split")
train_test_split = TrainTestSplit(train_features, train_label, test_size=0.3, stratify=train_label)
X_train, X_validation, y_train, y_validation = train_test_split.split()

# smote sampling
print("Over-sampling: SMOTE")
over_sampling = OverSamplingTask(X_train, y_train)
X_resampled, y_resampled = over_sampling.smote()
#
# Train Validation -
print("Train Validation process")
#
# base_estimators = ["nearest_neighbors", "logistic_regression", "decision_tree", "random_forest"]
# for classifier in base_estimators:
#     print("Classifier: {0}".format(classifier))
#     save_path = get_path_save(classifier, estimator=None)
#     train_validation = None
#     if classifier == "nearest_neighbors":
#         train_validation = TrainValidationKNN(X_resampled, X_validation, y_resampled, y_validation, "euclidean", save_path)
#     elif classifier == "logistic_regression":
#         train_validation = TrainValidationLR(X_resampled, X_validation, y_resampled, y_validation, save_path)
#     elif classifier == "random_forest":
#         train_validation = TrainValidationRF(X_resampled, X_validation, y_resampled, y_validation, save_path)
#     elif classifier == "decision_tree":
#         train_validation = TrainValidationDT(X_resampled, X_validation, y_resampled, y_validation, save_path)
#     train_validation.run()
# #
# #
# classifier = "one_vs_rest"
# print("Classifier: {0}".format(classifier))
# for type_classifier in base_estimators:
#     print("Base estimator: {0}".format(type_classifier))
#     save_path = get_path_save(classifier, estimator=type_classifier)
#     train_validation = TrainValidationOVR(X_resampled, X_validation, y_resampled, y_validation, type_classifier, save_path, metric="euclidean")
#     train_validation.run()
#
# # classifier = "ada_boosting"
# # print("Classifier: {0}".format(classifier))
# # for type_classifier in ["random_forest"]:
# #     print("Base estimator: {0}".format(type_classifier))
# #     save_path = get_path_save(classifier, pearson_option, estimator=type_classifier)
# #     train_validation = TrainValidationAB(X_resampled, X_validation, y_resampled, y_validation, type_classifier, save_path, metric="euclidean")
# #     train_validation.run()
#
# classifier = "xgboost"
# print("Classifier: {0}".format(classifier))
# save_path = get_path_save(classifier)
# train_validation = TrainValidationXGBoost(X_resampled, X_validation.as_matrix(), y_resampled, y_validation, save_path)
# train_validation.run()

classifier = "bagging"
print("Classifier: {0}".format(classifier))
for type_classifier in ["random_forest"]:
    print("Base estimator: {0}".format(type_classifier))
    save_path = get_path_save(classifier, estimator=type_classifier)
    train_validation = TrainValidationBagging(X_resampled, X_validation, y_resampled, y_validation, type_classifier, save_path, metric="euclidean")
    train_validation.run()