
def get_grid_parameters():
    return [
        {"penalty": ["l2"],
         "dual": [True, False],
         "C": [100.0, 10.0, 1.0, 0.1, 0.01],
         "fit_intercept": [True, False]},
        {"penalty": ["l1"],
         "dual": [False],
         "C": [1e-7, 4.0, 2, 4.0/3.0, 1.0],
         "fit_intercept": [True, False]},
    ]
