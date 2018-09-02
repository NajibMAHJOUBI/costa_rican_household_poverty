# coding: utf-8
# https://chrisalbon.com/machine_learning/feature_selection/drop_highly_correlated_features/

import numpy as np


class PearsonSelectorTask:

    def __init__(self, data, id_column, label_column, continuous_features, correlation):
        self.__data__ = data
        self.__id_column__ = id_column
        self.__label_column__ = label_column
        self.__continuous_features__ = continuous_features
        self.correlation = correlation

    def __str__(self):
        s = "Pearson Selector"
        return s

    def compute_correlation_matrix(self):
        # Create correlation matrix
        return self.__data__[self.__continuous_features__].corr().abs()

    def define_restrained_features(self):
        # Select upper triangle of correlation matrix
        corr_matrix = self.compute_correlation_matrix()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

        # Find index of feature columns with correlation greater than 0.95
        to_drop = [column for column in upper.columns if any(upper[column] > self.correlation)]
        print(to_drop)
        for column in to_drop:
            self.__continuous_features__.remove(column)

    def get_restrained_features(self, data, option):
        if option == "train":
            return data[self.__continuous_features__ + [self.__id_column__, self.__label_column__]]
        elif option == "test":
            return data[self.__continuous_features__ + [self.__id_column__]]

    def get_restrained_columns(self):
        return self.__continuous_features__
