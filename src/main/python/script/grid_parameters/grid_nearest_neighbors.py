
def get_grid_parameters(metric):
    # grid_parameters = [{"n_neighbors": [1, 5, 10, 15],
    #                     "weights": ["uniform", "distance"],
    #                     "algorithm": ["brute"],
    #                     "metric": [metric],
    #                     "leaf_size": [30]}]
    #
    # if metric != "jaccard":
    #     grid_parameters.append({"n_neighbors": [1, 5, 10, 15],
    #                             "weights": ["uniform", "distance"],
    #                             "algorithm": ["ball_tree"],
    #                             "leaf_size": [10, 20, 30],
    #                             "metric": [metric]})
    grid_parameters = [{"n_neighbors": [1, 5, 10, 15],
                        "weights": ["uniform", "distance"],
                        "algorithm": ["brute"],
                        "metric": [metric],
                        "leaf_size": [30]},
                       {"n_neighbors": [1, 5, 10, 15],
                        "weights": ["uniform", "distance"],
                        "algorithm": ["ball_tree"],
                        "leaf_size": [10, 20, 30],
                        "metric": [metric]}]
    return grid_parameters
