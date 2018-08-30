# coding: utf-8

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def all_scores(y_true, y_prediction, average_option):
    accuracy = accuracy_score(y_true, y_prediction)
    precision = precision_score(y_true, y_prediction, average=average_option)
    recall = recall_score(y_true, y_prediction, average=average_option)
    f1 = f1_score(y_true, y_prediction, average=average_option)
    return accuracy, precision, recall, f1
