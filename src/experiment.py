"""Experiment diagnostics and uncertainty, separate from predictive modeling."""

import numpy as np
import pandas as pd
from scipy.stats import chisquare, norm

from src.data import ARMS, PRE_TREATMENT


def ate(y, treatment):
    y, t = np.asarray(y, dtype=float), np.asarray(treatment)
    if len(y) != len(t) or not np.isin(t, [0, 1]).all() or not np.isfinite(y).all():
        raise ValueError("Finite outcomes and aligned binary treatment required")
    control, treated = y[t == 0], y[t == 1]
    if min(len(control), len(treated)) < 2:
        raise ValueError("ATE requires at least two observations in each arm")
    effect = treated.mean() - control.mean()
    se = np.sqrt(treated.var(ddof=1) / len(treated) + control.var(ddof=1) / len(control))
    margin = norm.ppf(0.975) * se
    return {
        "control_rate": float(control.mean()),
        "treatment_rate": float(treated.mean()),
        "ate": float(effect),
        "ci_low": float(effect - margin),
        "ci_high": float(effect + margin),
        "n_control": len(control),
        "n_treatment": len(treated),
    }


def randomization_check(frame):
    counts = frame.segment.value_counts().reindex(ARMS, fill_value=0)
    stat, p = chisquare(counts, np.repeat(len(frame) / 3, 3))
    return {
        "counts": counts.to_dict(),
        "srm_chi2": float(stat),
        "srm_p_value": float(p),
        "expected_allocation": "1:1:1",
        "srm_flag": bool(p < 0.01),
    }


def balance_table(frame):
    rows = []
    for col in PRE_TREATMENT:
        if pd.api.types.is_numeric_dtype(frame[col]):
            a, b = frame.loc[frame.treatment == 1, col], frame.loc[frame.treatment == 0, col]
            scale = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
            rows.append(
                {"feature": col, "smd": float((a.mean() - b.mean()) / scale) if scale else 0.0}
            )
        else:
            for level in sorted(frame[col].unique()):
                a = (frame.loc[frame.treatment == 1, col] == level).astype(float)
                b = (frame.loc[frame.treatment == 0, col] == level).astype(float)
                scale = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
                rows.append(
                    {
                        "feature": f"{col}={level}",
                        "smd": float((a.mean() - b.mean()) / scale) if scale else 0.0,
                    }
                )
    return pd.DataFrame(rows)
