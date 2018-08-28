# coding: utf-8

import pandas as pd


def build_submission(id, prediction, path):
    d = {"Id": id, "Target": prediction}
    submission = pd.DataFrame(d)
    submission.to_csv(path, index=False)