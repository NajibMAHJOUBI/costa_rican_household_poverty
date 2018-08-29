
def get_grid_parameters():
    return [{"n_estimators": [10, 15, 20, 25],
             "criterion": ["gini", "entropy"],
             "bootstrap": [True, False],
             "max_depth": [2, 4, 6, 8, 10],
             "max_features": ["sqrt", "log2", None]}]
