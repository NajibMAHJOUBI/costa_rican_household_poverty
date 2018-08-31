
def get_grid_parameters():
    return [{"n_neighbors": [1, 5, 10, 15], "weights": ["uniform", "distance"],
             "algorithm": ["ball_tree"], "leaf_size": [10, 20, 30],
             "metric": ["jaccard"]},
            {"n_neighbors": [1, 5, 10, 15], "weights": ["uniform", "distance"],
             "algorithm": ["brute"], "metric": ["jaccard"], "leaf_size": [30]}]
