
def get_grid_parameters(metric):
    return [{"n_neighbors": [1, 5, 10, 15], "weights": ["uniform", "distance"],
             "algorithm": ["ball_tree"], "leaf_size": [10, 20, 30],
             "metric": [metric]},
            {"n_neighbors": [1, 5, 10, 15], "weights": ["uniform", "distance"],
             "algorithm": ["brute"], "metric": [metric], "leaf_size": [30]}]
