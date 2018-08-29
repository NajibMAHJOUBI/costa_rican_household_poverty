# coding: utf-8

from sklearn.model_selection import train_test_split


class TrainTestSplit:

    def __init__(self, features, label, test_size=0.2, stratify=None):
        self.__features__ = features
        self.__label__ = label
        self.__test_size__ = test_size
        self.__stratify__ = stratify

    def __str__(self):
        s = "train test split"
        return s

    def split(self):
        return train_test_split(self.__features__, self.__label__, test_size=self.__test_size__,
                                stratify=self.__stratify__)
