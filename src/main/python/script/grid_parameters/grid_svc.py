
def get_grid_parameters():
    list_C = [0.01, 0.1, 1.0, 10.0, 100.0]
    list_gamma = [1.0, 10.0, 100.0, "auto"]
    return [{"kernel": ["linear"], "degree": [3], "gamma": ["auto"], "C": list_C},
            {"kernel": ["rbf"], "degree": [3], "gamma": list_gamma, "C": list_C},
            {"kernel": ["sigmoid"], "degree": [3], "gamma": list_gamma, "C": list_C},
            {"kernel": ["poly"], "degree": [2, 3], "gamma": list_gamma,  "C": list_C}]
