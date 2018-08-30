
from sklearn.preprocessing import StandardScaler


class StandardScalerTask:

    def __init__(self, data):
        self.data = data

    def __str__(self):
        s = "Standard Scaler Task"
        return s

    def define_estimator(self):
        self.estimator = StandardScaler(copy=True, with_mean=True, with_std=True)

    def fit(self):
        self.model = self.estimator.fit(self.data)

    def transform(self):
        return self.model.transform(self.data)
