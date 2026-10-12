"""Load the public randomized experiment without redistributing customer rows."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklift.datasets import fetch_hillstrom

ARMS = ["Mens E-Mail", "Womens E-Mail", "No E-Mail"]
PRE_TREATMENT = [
    "recency",
    "history_segment",
    "history",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
]


def validate_data(frame):
    required = PRE_TREATMENT + ["segment", "visit", "conversion", "spend"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if frame[required].isna().any().any():
        raise ValueError("Missing data: inspect source before analysis; no silent imputation.")
    if set(frame.segment.unique()) != set(ARMS):
        raise ValueError("Expected all three randomized experiment arms.")
    for col in ["visit", "conversion", "mens", "womens", "newbie"]:
        if not frame[col].isin([0, 1]).all():
            raise ValueError(f"{col} must be binary")
    for col in ["recency", "history", "spend"]:
        if not np.isfinite(frame[col]).all() or (frame[col] < 0).any():
            raise ValueError(f"Invalid numeric values in {col}")
    if (frame.conversion > frame.visit).any():
        raise ValueError("Conversion without a visit: outcome inconsistency")
    return {
        "rows": len(frame),
        "missing_cells": int(frame[required].isna().sum().sum()),
        "duplicate_profiles": int(frame[PRE_TREATMENT].duplicated().sum()),
        "duplicate_full_rows": int(frame[required].duplicated().sum()),
        "note": "No customer ID: identical profiles are flagged, never automatically removed.",
    }


def load_data(cache=Path("data/raw"), csv=None):
    if csv:
        frame = pd.read_csv(csv)
    else:
        dataset = fetch_hillstrom(target_col="all", data_home=str(cache))
        frame = pd.concat([dataset.data, dataset.treatment, dataset.target], axis=1)
    frame = frame.reset_index(drop=True)
    validate_data(frame)
    return frame


def binary_experiment(frame):
    result = frame.loc[frame.segment.isin(["Mens E-Mail", "No E-Mail"])].copy()
    result["treatment"] = (result.segment == "Mens E-Mail").astype(int)
    # Original row index is a reproducibility key, never a model input.
    result.index.name = "row_id"
    return result


def split_data(frame, seed=42):
    """60/20/20, stratified by assignment only; outcomes do not determine splits."""
    train, rest = train_test_split(
        frame, test_size=0.4, random_state=seed, stratify=frame.treatment
    )
    validation, test = train_test_split(
        rest, test_size=0.5, random_state=seed + 1, stratify=rest.treatment
    )
    return train, validation, test
