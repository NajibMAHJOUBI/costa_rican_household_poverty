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
from classification.random_forest import RandomForestTask
from classification.one_vs_rest import OneVsRestTask
import pandas as pd
from standard_scaler.standard_scaler import StandardScalerTask


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
pearson_option = False
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
test_features = scaler.transform(test_features)

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

estimator = OneVsRestTask()
base_estimator = RandomForestTask(n_estimators=50,
                                  criterion="gini",
                                  max_depth=15,
                                  min_samples_split=10,
                                  min_samples_leaf=5)
base_estimator.define_estimator()
estimator.set_classifier(base_estimator.get_estimator())
estimator.define_estimator()
estimator.fit(X_resampled, y_resampled)
prediction = estimator.predict(test_features)

# estimator = XGBoostClassifierTask(max_depth=30, n_estimators=200)
# estimator.define_estimator()
# estimator.fit(X_resampled, y_resampled)
# prediction = estimator.predict(test_features)

d = {"Id": test_id, "Target": prediction}
data = pd.DataFrame(d)
path = os.path.join(os.path.realpath("../../../../../.."), "submission/sklearn/train_validation/continuous/one_vs_rest/random_forest/submission/prediction.csv")
data.to_csv(path, index=False)


