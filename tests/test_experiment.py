import numpy as np
import pandas as pd
import pytest

from src.data import PRE_TREATMENT, split_data
from src.experiment import ate, randomization_check
from src.features import make_features


def test_ate_known_effect_and_no_effect():
    result = ate([0, 0, 1, 1, 0, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1])
    assert result["ate"] == 0.25
    assert result["ci_low"] < 0.25 < result["ci_high"]
    assert ate([0, 1, 0, 1], [0, 0, 1, 1])["ate"] == 0


def test_ate_rejects_missing_arm():
    with pytest.raises(ValueError):
        ate([0, 1], [1, 1])


def test_srm_flags_wrong_allocation():
    balanced = pd.DataFrame({"segment": ["Mens E-Mail", "Womens E-Mail", "No E-Mail"] * 100})
    assert randomization_check(balanced)["srm_p_value"] == 1
    broken = pd.DataFrame({"segment": ["Mens E-Mail"] * 200 + ["No E-Mail"] * 100})
    assert randomization_check(broken)["srm_flag"]


def test_splits_disjoint_complete_reproducible():
    frame = pd.DataFrame({"treatment": [0, 1] * 100})
    parts = split_data(frame)
    assert sorted(np.concatenate([p.index for p in parts])) == list(frame.index)
    assert not set(parts[0].index) & set(parts[1].index)
    assert not set(parts[0].index) & set(parts[2].index)
    assert not set(parts[1].index) & set(parts[2].index)
    assert all(a.index.equals(b.index) for a, b in zip(parts, split_data(frame)))


def test_post_treatment_changes_cannot_change_features():
    frame = pd.DataFrame({c: [1, 2] for c in PRE_TREATMENT})
    frame["visit"], frame["conversion"], frame["spend"], frame["treatment"] = 0, 0, 0, 0
    before = make_features(frame)
    frame[["visit", "conversion", "spend", "treatment"]] = 999
    pd.testing.assert_frame_equal(before, make_features(frame))
