"""Standard sklift metrics, with their units and definitions retained in reports."""

import numpy as np
from sklift.metrics import qini_auc_score, qini_curve, uplift_at_k, uplift_auc_score


def ranking_metrics(y, treatment, scores):
    if min(np.sum(np.asarray(treatment) == 0), np.sum(np.asarray(treatment) == 1)) < 2:
        raise ValueError("Ranking metrics require both experiment arms")
    result = {}
    for name, prediction in scores.items():
        result[name] = {
            "qini_auc_normalized": float(qini_auc_score(y, prediction, treatment)),
            "auuc_normalized": float(uplift_auc_score(y, prediction, treatment)),
            "uplift_at_20pct": float(
                uplift_at_k(y, prediction, treatment, k=0.2, strategy="overall")
            ),
        }
    return result


def get_qini(y, treatment, scores):
    x, gain = qini_curve(y, scores, treatment)
    return x / len(y), gain
