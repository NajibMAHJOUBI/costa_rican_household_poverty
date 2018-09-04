
from sklearn.preprocessing import StandardScaler


class StandardScalerTask:

    def __init__(self, data):
        self.data = data
        self.estimator = None
        self.model = None

    def __str__(self):
        s = "Standard Scaler Task"
        return s

    def define_estimator(self):
        self.estimator = StandardScaler(copy=True,
                                        with_mean=True,
                                        with_std=True)

    def fit(self):
        self.model = self.estimator.fit(self.data)

    def transform(self, data):
        return self.model.transform(data)
