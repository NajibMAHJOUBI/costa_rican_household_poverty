
def get_grid_parameters():
    list_C = [0.01, 0.1, 1.0, 10.0, 100.0]
    list_gamma = [1.0, 10.0, 100.0, "auto"]
    return [{"kernel": ["linear"], "C": list_C},
            {"kernel": ["rbf"], "gamma": list_gamma, "C": list_C},
            {"kernel": ["sigmoid"], "gamma": list_gamma, "C": list_C},
            {"kernel": ["poly"],  "degree": [2, 3],  "gamma": list_gamma,  "C": list_C}
            ]
