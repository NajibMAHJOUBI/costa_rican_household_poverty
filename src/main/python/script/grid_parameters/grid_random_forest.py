
def get_grid_parameters():
    return [{"n_estimators": [50, 100, 150],
             "criterion": ["gini"],
             "max_depth": [5, 10, 15],
             "min_samples_split": [5, 10, 15],
             "min_samples_leaf": [5, 10, 15]}]
