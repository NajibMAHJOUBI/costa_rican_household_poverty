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
from train_validation.train_validation_knn import TrainValidationKNN
from train_validation.train_validation_lr import TrainValidationLR
from train_validation.train_validation_rf import TrainValidationRF
from train_validation.train_validation_ovr import TrainValidationOVR
from train_validation.train_validation_dt import TrainValidationDT


# def utils function
def get_path_save(classifier, selector_option, estimator=None):
    path_classifier = None
    if estimator is not None:
        path_classifier = "{0}/{1}".format(classifier, estimator)
    else:
        path_classifier = classifier
    path = os.path.join(os.path.realpath("../../../../../.."), "submission/sklearn/train_validation/continuous")
    if selector_option:
        return os.path.join(path, "{0}/pearson_features".format(path_classifier))
    else:
        return os.path.join(path, "{0}/all_features".format(path_classifier))

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

classifier = "one_vs_rest"

save_path = get_path_save(classifier, pearson_option, estimator=None)
train_validation = None
if classifier == "nearest_neighbors":
    train_validation = TrainValidationKNN(X_resampled, X_validation, y_resampled, y_validation, "euclidean", save_path)
elif classifier == "logistic_regression":
    train_validation = TrainValidationLR(X_resampled, X_validation, y_resampled, y_validation, save_path)
elif classifier == "random_forest":
    train_validation = TrainValidationRF(X_resampled, X_validation, y_resampled, y_validation, save_path)
elif classifier == "decision_tree":
    train_validation = TrainValidationDT(X_resampled, X_validation, y_resampled, y_validation, save_path)
elif classifier == "one_vs_rest":
    type_classifier = "random_forest"
    save_path = get_path_save(classifier, pearson_option, estimator=type_classifier)
    print(save_path)
    train_validation = TrainValidationOVR(X_resampled, X_validation, y_resampled, y_validation, type_classifier, save_path, metric="euclidean")


train_validation.run()



