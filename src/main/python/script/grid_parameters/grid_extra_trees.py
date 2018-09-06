
def get_grid_parameters():
    return [{"n_estimators": [100, 200, 300],
             "criterion": ["gini"],
             "max_depth": [10, 20, 30],
             "min_samples_split": [2, 3, 4],
             "min_samples_leaf": [2, 3, 4]}]
