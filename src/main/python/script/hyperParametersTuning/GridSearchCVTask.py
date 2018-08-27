
from classification.KNeighborsClassifierTask import KNeighborsClassifierTask
from gridParameters import gridKNearestNeighbors
from sklearn.model_selection import GridSearchCV


class GridSearchCVTask:

    def __init__(self, number_folds, estimator_algorithm):
        self.number_folds = number_folds
        self.estimator_algorithm = estimator_algorithm

    def __str__(self):
        pass

    def estimator(self):
        if self.estimator_algorithm == "kNearestNeighbors":
            return KNeighborsClassifierTask().define_estimator().get_estimator()

    def grid_search_cv(self):
        self.grid = GridSearchCV(self.estimator, cv=self.number_folds, param_grid=self.grid_parameters(), verbose=1)

    def grid_parameters(self):
        if self.estimator_algorithm == "kNearestNeighbors":
            return gridKNearestNeighbors.get_grid_parameters()

    def fit(self, X, y):
        self.model = self.grid.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


if __name__ == "__main__":
    from sklearn import datasets
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = iris.target

    print("{0}".format("test"))
    search_cv = GridSearchCVTask(3, "kNearestNeighbors")
    search_cv.grid_search_cv()
    search_cv.fit(X, y)

    print(search_cv.predict(X))
    print(type(X))

