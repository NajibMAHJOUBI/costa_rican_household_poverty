# coding: utf-8

import os
import sys
sys.path.append(os.path.realpath('../..'))

# import libraries
from utils.load_data_task import LoadDataTask
from utils.define_label_features import DefineLabelFeaturesTask
from over_sampling.over_sampling_task import OverSamplingTask
from hyperParametersTuning.train_validation_rf import TrainValidationRF
from split_task.train_test_split import TrainTestSplit
from features_selector.chi2_selector import Chi2SelectorTask

# Load datasets
path_data = "../../../../../../data"
# --> train
load_data_task = LoadDataTask(os.path.join(path_data, "train/train.csv"))
train = load_data_task.load_data()
# --> test
load_data_task = LoadDataTask(os.path.join(path_data, "test/test.csv"))
test = load_data_task.load_data()

print("Train shape: {0}".format(train.shape))
print("Test shape: {0}".format(test.shape))

# Features selection
categorical_features = open("../../../../resources/categoricalFeatures").read().split(",")
chi2_option = True
if chi2_option:
    # Chi-square selection
    chi2_selector = Chi2SelectorTask(train, "Id", "Target", categorical_features, 0.05)
    chi2_selector.define_restrained_features()
    train = chi2_selector.get_restrained_features(train, "train")
    test = chi2_selector.get_restrained_features(test, "test")
    categorical_features = chi2_selector.get_restrained_columns()
    print("Train shape: {0}".format(train.shape))
    print("Test shape: {0}".format(test.shape))

# define features and label
define_label_features = DefineLabelFeaturesTask("Id", "Target", categorical_features)
train_id = define_label_features.get_id(train)
train_label  = define_label_features.get_label(train)
train_features = define_label_features.get_features(train)

test_id = define_label_features.get_id(test)
test_features = define_label_features.get_features(test)

# train-validation split
train_test_split = TrainTestSplit(train_features, train_label, test_size=0.3, stratify=train_label)
X_train, X_validation, y_train, y_validation = train_test_split.split()

# smote sampling
over_sampling = OverSamplingTask(X_train, y_train)
X_resampled, y_resampled = over_sampling.smote()

# Train Validation Random Forest classifier
save_path = None
if chi2_option:
    save_path = os.path.join(os.path.realpath("../../../../../.."), "submission/sklearn/train_validation/categorical/chi2_features")
else:
    save_path = os.path.join(os.path.realpath("../../../../../.."), "submission/sklearn/train_validation/categorical/all_features")

train_validation = TrainValidationRF(X_resampled, X_validation, y_resampled, y_validation, save_path)
train_validation.run()
