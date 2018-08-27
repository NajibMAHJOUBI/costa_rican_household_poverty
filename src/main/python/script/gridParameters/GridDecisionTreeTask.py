
def get_grid_parameters():
    return [{"criterion": ["gini", "entropy"],
             "splitter": ["best", "random"],
             "max_depth": [2, 4, 6, 8, 10],
             "max_features": ["sqrt", "log2", None]}]
