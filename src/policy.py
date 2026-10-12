"""Evaluate contact policies using randomized assignments, never predicted uplift sums."""

import numpy as np
from scipy.stats import norm


def top_k(scores, fraction, seed=42):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all() or not 0 <= fraction <= 1:
        raise ValueError("Finite 1D scores and fraction in [0,1] required")
    # Break ties independently of outcomes and treatment; exactly floor(n * fraction).
    tie = np.random.default_rng(seed).random(len(scores))
    order = np.lexsort((tie, -scores))
    mask = np.zeros(len(scores), dtype=bool)
    mask[order[: int(len(scores) * fraction)]] = True
    return mask


def pseudo_outcome(y, treatment, propensity=0.5):
    y, t = np.asarray(y, dtype=float), np.asarray(treatment)
    if (
        y.ndim != 1
        or t.shape != y.shape
        or len(y) < 2
        or not np.isfinite(y).all()
        or not np.isin(t, [0, 1]).all()
        or not 0 < propensity < 1
    ):
        raise ValueError("Aligned finite outcomes, binary assignment and 0 < p < 1 required")
    return t * y / propensity - (1 - t) * y / (1 - propensity)


def net_contributions(y, treatment, mask, value=50.0, cost=0.1, propensity=0.5):
    """Incremental net value per eligible customer vs never contacting anyone."""
    mask = np.asarray(mask)
    if mask.shape != np.asarray(y).shape or not np.isin(mask, [0, 1]).all():
        raise ValueError("Aligned binary policy required")
    if not np.isfinite([value, cost]).all() or value < 0 or cost < 0:
        raise ValueError("Value and cost must be finite and nonnegative")
    return mask * (value * pseudo_outcome(y, treatment, propensity) - cost)


def estimate(contributions):
    z = np.asarray(contributions, dtype=float)
    if z.ndim != 1 or len(z) < 2 or not np.isfinite(z).all():
        raise ValueError("At least two finite contributions required")
    mean = float(z.mean())
    se = float(z.std(ddof=1) / np.sqrt(len(z)))
    margin = norm.ppf(0.975) * se
    return {
        "net_per_customer": mean,
        "ci_low": mean - margin,
        "ci_high": mean + margin,
        "net_per_1000": mean * 1000,
    }


def evaluate_policy(y, treatment, mask, value=50.0, cost=0.1):
    z = net_contributions(y, treatment, mask, value, cost)
    incremental = np.asarray(mask) * pseudo_outcome(y, treatment)
    return {
        **estimate(z),
        "coverage": float(np.mean(mask)),
        "incremental_conversions_per_1000": float(incremental.mean() * 1000),
    }


def choose_policy(scores, y, treatment, fractions, value, cost):
    """Validation only. Simple alternatives compete with ML on the same objective."""
    candidates = [
        ("none", 0.0, np.zeros(len(y), dtype=bool)),
        ("all", 1.0, np.ones(len(y), dtype=bool)),
    ]
    for fraction in fractions:
        candidates.append(("random", fraction, top_k(np.zeros(len(y)), fraction)))
        for name, prediction in scores.items():
            candidates.append((name, fraction, top_k(prediction, fraction)))
    rows = []
    for name, fraction, mask in candidates:
        rows.append(
            {
                "policy": name,
                "fraction": fraction,
                **evaluate_policy(y, treatment, mask, value, cost),
            }
        )
    # Declared selection objective: highest point estimate; exploratory, no CI optimization.
    best = max(rows, key=lambda row: row["net_per_customer"])
    return best, rows
