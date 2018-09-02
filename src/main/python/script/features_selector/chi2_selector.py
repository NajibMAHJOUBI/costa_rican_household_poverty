
from sklearn import feature_selection


class Chi2SelectorTask:

    def __init__(self, data, id_column, label_column, categorical_features, alpha):
        self.__data__ = data
        self.__id_column__ = id_column
        self.__label_column__ = label_column
        self.__categorical_features__ = categorical_features
        self.__alpha__ = alpha
        self.restrained_features = None

    def __str__(self):
        s = "Chi2 Selector Task"
        return s

    def get_p_value(self):
        X = self.__data__[self.__categorical_features__]
        y = self.__data__[self.__label_column__]
        chi2, p_value = feature_selection.chi2(X, y)
        return p_value

    def define_restrained_features(self):
        self.restrained_features = [self.__categorical_features__[ind]
                                      for ind, value in enumerate(self.get_p_value() < self.__alpha__) if value]

    def get_restrained_features(self, data, option):
        if option == "train":
            return data[self.restrained_features + [self.__id_column__, self.__label_column__]]
        elif option == "test":
            return data[self.restrained_features + [self.__id_column__]]

    def get_restrained_columns(self):
        return self.restrained_features
