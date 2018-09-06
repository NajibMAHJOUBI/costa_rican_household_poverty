
from sklearn.externals import joblib


class ClassifierTask:

    def __init__(self):
        self.estimator = None
        self.model = None
        self.best_model = None
        self.best_score = -1.0 * float("inf")

    def __str__(self):
        s = "Upper Classifier Class"
        return s

    def fit(self, features, label):
        self.model = self.estimator.fit(features, label)

    def predict(self, features):
        return self.model.predict(features)

    def get_estimator(self):
        return self.estimator

    def define_best_model(self, model, score):
        if score > self.best_model:
            self.best_model = score
            self.best_model = model

    def save_best_model(self):
        joblib.dump(self.best_model, 'filename.pkl')
