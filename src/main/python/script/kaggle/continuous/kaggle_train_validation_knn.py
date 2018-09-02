# coding: utf-8

import os
import sys
sys.path.append(os.path.realpath('../..'))

# import libraries
from utils.load_data_task import LoadDataTask
from utils.define_label_features import DefineLabelFeaturesTask
from over_sampling.over_sampling_task import OverSamplingTask
from train_validation.train_validation_knn import TrainValidationKNN
from split_task.train_test_split import TrainTestSplit
from features_selector.pearson_selector import PearsonSelectorTask
from fill_na.fill_na import FillNaValuesTask

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

# define features and label
define_label_features = DefineLabelFeaturesTask("Id", "Target", continuous_features)
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

# Train Validation kNN
save_path = None
if pearson_option:
    save_path = os.path.join(os.path.realpath("../../../../../.."), "submission/sklearn/train_validation/continuous/nearest_neighbors/pearson_features")
else:
    save_path = os.path.join(os.path.realpath("../../../../../.."), "submission/sklearn/train_validation/continuous/nearest_neighbors/all_features")

train_validation = TrainValidationKNN(X_resampled, X_validation, y_resampled, y_validation, "euclidean", save_path)
train_validation.run()
