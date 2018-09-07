# coding: utf-8

class ClassifierTask:

    def __init__(self):
        self.estimator = None
        self.model = None

    def __str__(self):
        s = "Upper Classifier Class"
        return s

    def fit(self, features, label):
        self.model = self.estimator.fit(features, label)

    def predict(self, features):
        return self.model.predict(features)

    def get_estimator(self):
        return self.estimator

    def get_model(self):
        return self.model
