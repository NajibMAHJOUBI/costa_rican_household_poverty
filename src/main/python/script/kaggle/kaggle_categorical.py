import os
import sys
sys.path.append(os.path.realpath('..'))

from utils.load_data_task import LoadDataTask
from utils.define_label_features import DefineLabelFeaturesTask
from classification.random_forest import RandomForestTask
from classification.decision_tree import DecisionTreeTask
from classification.nearest_neighbors import KNeighborsClassifierTask
from classification.gaussian_nb import GaussianNBTask
from utils.build_submission import build_submission

path_data = "../../../../../data"
load_data_task = LoadDataTask(os.path.join(path_data, "train/train.csv"))
train = load_data_task.load_data()

load_data_task = LoadDataTask(os.path.join(path_data, "test/test.csv"))
test = load_data_task.load_data()



define_label_features = DefineLabelFeaturesTask("Id", "Target",
                                        open("../../../resources/categoricalFeatures").read().split(","))
train_id = define_label_features.get_id(train)
train_label  = define_label_features.get_label(train)
train_features = define_label_features.get_features(train)

test_id = define_label_features.get_id(test)
test_features = define_label_features.get_features(test)

classifier_list = ["gaussian_nb"]  # "decision_tree", "random_forest", "nearest_neighbors"
algorithm = None
for classifier in classifier_list:
    print("Classifier: {0}".format(classifier))
    if classifier == "decision_tree":
        algorithm = DecisionTreeTask()
    elif classifier == "random_forest":
        algorithm = RandomForestTask()
    elif classifier == "nearest_neighbors":
        algorithm = KNeighborsClassifierTask()
    elif classifier == "gaussian_nb":
        algorithm = GaussianNBTask()

    algorithm.define_estimator()
    algorithm.fit(train_features, train_label)
    prediction = algorithm.predict(test_features)
    build_submission(test_id, prediction, "../../../../../submission/sklearn/classifier/{0}.csv".format(classifier))
