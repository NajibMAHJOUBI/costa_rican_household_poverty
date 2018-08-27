
class ClassifierTask:

    def __init__(self):
        self.estimator = None
        self.model = None

    def fit(self, X, y):
        self.model = self.estimator.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def get_estimator(self):
        return self.estimator
