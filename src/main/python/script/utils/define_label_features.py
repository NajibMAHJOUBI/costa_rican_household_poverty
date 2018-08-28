

class DefineLabelFeaturesTask:

    def __init__(self, id_column, label_column, feature_column):
        self.id_column = id_column
        self.label_column = label_column
        self.feature_column = feature_column

    def __str__(self):
        s = "Label column: {0}\n".format(self.label_column)
        s += "Feature columns: {0}\n".format(self.feature_column)
        return s

    def get_id(self, data):
        return data[self.id_column]

    def get_features(self, data):
        return data[self.feature_column]

    def get_label(self, data):
        return data[self.label_column]


if __name__ == "__main__":
    import pandas as pd
    d = {"id": ["a", "b", "c", "d", "e", "f"],
         "target": [0, 0, 0, 1, 1, 1],
         "x": [0, 0, 0, 0, 0, 0],
         "y": [1, 2, 3, 4, 5, 6]}
    data = pd.DataFrame(d)

    feature_columns = ["x", "y"]
    label_features = DefineLabelFeaturesTask("id", "target", feature_columns)
    id = label_features.get_id(data)
    features = label_features.get_features(data)
    label = label_features.get_label(data)
    assert(len(id.shape) == 1)
    assert(id.shape[0] == data.shape[0])
    assert(len(label.shape) == 1)
    assert(label.shape[0] == data.shape[0])
    assert(features.shape[0] == data.shape[0])
    assert(features.shape[1] == len(feature_columns))
