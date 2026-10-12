import numpy as np
import pytest

from src.policy import choose_policy, evaluate_policy, net_contributions, top_k


def test_ipw_recovers_known_incremental_value_in_balanced_pairs():
    # Identical potential outcomes in each pair; treated always buys, control never buys.
    y, t = [0, 1] * 100, [0, 1] * 100
    result = evaluate_policy(y, t, np.ones(200), value=10, cost=2)
    assert result["net_per_customer"] == 8
    assert result["incremental_conversions_per_1000"] == 1000
    assert evaluate_policy(y, t, np.zeros(200))["net_per_customer"] == 0


def test_targeting_excludes_harmed_group():
    y, t = [0, 1, 1, 0] * 100, [0, 1, 0, 1] * 100
    mask = np.array([1, 1, 0, 0] * 100)
    assert evaluate_policy(y, t, mask, value=10, cost=0)["net_per_customer"] == 5
    assert evaluate_policy(y, t, np.ones(400), value=10, cost=0)["net_per_customer"] == 0


def test_top_k_budget_ties_and_empty_policies():
    scores = np.ones(101)
    assert top_k(scores, 0.2).sum() == 20
    np.testing.assert_array_equal(top_k(scores, 0.2), top_k(scores, 0.2))
    assert not top_k(scores, 0).any()
    assert top_k(scores, 1).all()
    assert top_k([1, 5, 3, 2], 0.5).tolist() == [False, True, True, False]


def test_selection_can_choose_no_contact_with_costs():
    best, _ = choose_policy(
        {"model": np.arange(100)}, np.zeros(100), [0, 1] * 50, [0.2, 0.5], value=50, cost=1
    )
    assert best["policy"] == "none"


def test_paired_difference_identical_policy_is_zero():
    from src.policy import estimate

    z = net_contributions([0, 1, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0])
    result = estimate(z - z)
    assert result["ci_low"] == result["ci_high"] == 0


@pytest.mark.parametrize("fraction", [-0.1, 1.1, float("nan")])
def test_invalid_budget(fraction):
    with pytest.raises(ValueError):
        top_k([1, 2], fraction)


def test_invalid_assignment_and_economics():
    with pytest.raises(ValueError):
        net_contributions([1, 0], [2, 0], [1, 1])
    with pytest.raises(ValueError):
        net_contributions([1, 0], [1, 0], [1, 1], cost=-1)
