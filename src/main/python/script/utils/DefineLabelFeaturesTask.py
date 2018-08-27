

class DefineLabelFeaturesTask:

    def __init__(self, data, label_column, feature_column):
        self.data = data
        self.label_column = label_column
        self.feature_column = feature_column

    def __str__(self):
        s = "Data type: {0}".format(type(data))

    def get_x(self):
        return self.data[self.feature_column]

    def get_y(self):
        return self.data[self.label_column]


if __name__ == "__main__":
    import pandas as pd
    d = {"target": [0, 0, 0, 1, 1, 1], "x": [0, 0, 0, 0, 0, 0], "y": [1, 2, 3, 4, 5, 6]}
    data = pd.DataFrame(d)

    label_features = DefineLabelFeaturesTask(data, "target", ["x", "y"])




