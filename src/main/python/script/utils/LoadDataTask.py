import pandas as pd


class LoadDataTask:

    def __init__(self, path):
        self.path = path

    def __str__(self):
        s = "Path: {0}\n".format(self.path)

        return s

    def load_data(self):
        return pd.read_csv(self.path, sep=",", header=0)


if __name__ == "__main__":
    load_data_task = LoadDataTask("../../../../../data/train/train.csv")
    data = load_data_task.load_data()

    print(data.shape)
    print(data.describe())
