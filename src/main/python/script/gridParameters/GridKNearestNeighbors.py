
def get_grid_parameters():
    return [{"n_neighbors": [3, 5, 7, 9],
             "weights": ["uniform", "distance"],
             "algorithm": ["ball_tree", "kd_tree", "brute"],
             "leaf_size": [10, 20, 30],
             "p": [1, 2]}]
