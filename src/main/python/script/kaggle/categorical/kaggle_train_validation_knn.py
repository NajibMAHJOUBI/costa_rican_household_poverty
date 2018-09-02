# coding: utf-8

import os
import sys
sys.path.append(os.path.realpath('..'))

# import libraries
from utils.load_data_task import LoadDataTask
from utils.define_label_features import DefineLabelFeaturesTask
from over_sampling.over_sampling_task import OverSamplingTask
from hyperParametersTuning.train_validation_knn import TrainValidationKNN
from split_task.train_test_split import TrainTestSplit

# import datasets
path_data = "../../../../../data"
load_data_task = LoadDataTask(os.path.join(path_data, "train/train.csv"))
train = load_data_task.load_data()

load_data_task = LoadDataTask(os.path.join(path_data, "test/test.csv"))
test = load_data_task.load_data()

# define features and label
categorical_features = open("../../../resources/categoricalFeatures").read().split(",")
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

# Train Validation SVC classifier
save_path = os.path.join(os.path.realpath("../../../../.."), "submission/sklearn/train_validation/knn.csv")
train_validation = TrainValidationKNN(X_resampled, X_validation,
                                          y_resampled, y_validation,
                                          save_path)
train_validation.run()
