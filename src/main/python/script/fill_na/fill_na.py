
import numpy as np


class FillNaValuesTask:
    def __init__(self, train, test, continuous_column, option):
        self.train = train
        self.test = test
        self.continuous_column= continuous_column
        self.option = option

    def __str__(self):
        s = "Fill Nan values"
        return s

    def fill_by_values(self):
        features_values = None
        if self.option == "mean":
            features_values = self.train.mean()
        elif self.option == "median":
            features_values = self.train.median()

        train_null_columns = self.train.columns[self.train.isnull().any()].tolist()
        for column in train_null_columns:
            filled_function = lambda row: features_values[column] if np.isnan(row[column]) else row[column]
            self.train["new_{0}".format(column)] = self.train.apply(lambda row: filled_function(row), axis=1)
        self.train = self.train.drop(train_null_columns, axis=1)
        self.train = self.train.rename(index=str, columns={"new_{0}".format(column): column for column in train_null_columns})

        test_null_columns = self.test.columns[self.test.isnull().any()].tolist()
        for column in test_null_columns:
            filled_function = lambda row: features_values[column] if np.isnan(row[column]) else row[column]
            self.test["new_{0}".format(column)] = self.test.apply(lambda row: filled_function(row), axis=1)
        self.test = self.test.drop(test_null_columns, axis=1)
        self.test = self.test.rename(index=str, columns={"new_{0}".format(column): column for column in test_null_columns})

    def fill_yes_no(self, yes_no_columns):
        def filled_function(value):
            if value == "yes":
                return 1
            elif value == "no":
                return 0
            else:
                return value

        for column in yes_no_columns:
            self.train["new_{0}".format(column)] = self.train.apply(lambda row: filled_function(row[column]), axis=1)
            self.test["new_{0}".format(column)] = self.test.apply(lambda row: filled_function(row[column]), axis=1)

        self.train = self.train.drop(yes_no_columns, axis=1)
        self.test = self.test.drop(yes_no_columns, axis=1)

        columns = {"new_{0}".format(column): column for column in yes_no_columns}
        self.train = self.train.rename(index=str, columns=columns)
        self.test = self.test.rename(index=str, columns=columns)

    def get_train(self):
        return self.train

    def get_test(self):
        return self.test

