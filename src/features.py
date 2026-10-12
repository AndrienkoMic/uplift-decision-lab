"""Whitelist pre-treatment inputs; stateless transforms cannot learn from holdout."""

import numpy as np

from src.data import PRE_TREATMENT

CATEGORICAL = ["history_segment", "zip_code", "channel"]


def make_features(frame):
    x = frame[PRE_TREATMENT].copy()
    for col in CATEGORICAL:
        x[col] = x[col].astype(str)
    x["log_history"] = np.log1p(x.history)
    x["history_per_recency"] = x.history / (x.recency + 1)
    x["bought_both_categories"] = x.mens * x.womens
    return x
