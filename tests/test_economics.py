import numpy as np
import pytest

from src.economics import break_even_table, evaluate_scenarios, select_scenarios
from src.policy import choose_policy, evaluate_policy, top_k


def fixture():
    # High-uplift group: +1 purchase. Lower group: +0.5, in balanced treatment pairs.
    y = np.array([0, 1] * 100 + [0, 1, 1, 1] * 50)
    t = np.array([0, 1] * 200)
    scores = {"s_learner": np.array([1] * 200 + [0] * 200)}
    _, candidates = choose_policy(scores, y, t, [0.5], value=10, cost=0)
    return y, t, scores, candidates


def test_map_changes_from_all_to_targeting_to_none_at_known_break_even_costs():
    _, _, _, candidates = fixture()
    plans = select_scenarios(candidates, [10], [0, 6, 11])
    assert plans.policy.tolist() == ["all", "s_learner", "none"]
    assert plans.fraction.tolist() == [1, 0.5, 0]
    assert plans.validation_net_per_1000.tolist() == [7500, 2000, 0]


def test_zero_economics_ties_choose_none_and_axes_are_sorted_deduplicated():
    _, _, _, candidates = fixture()
    plans = select_scenarios(candidates, [10, 0, 10], [0, 0])
    assert len(plans) == 2
    assert plans.iloc[0].policy == "none"
    assert plans.value.tolist() == [0, 10]


def test_holdout_outcomes_change_values_but_cannot_change_locked_plans():
    y, t, scores, candidates = fixture()
    plans = select_scenarios(candidates, [10], [6])
    a = evaluate_scenarios(plans, scores, y, t)
    b = evaluate_scenarios(plans, scores, 1 - y, t)
    assert a.policy.tolist() == b.policy.tolist() == plans.policy.tolist()
    assert a.fraction.tolist() == b.fraction.tolist() == plans.fraction.tolist()
    assert a.test_net_per_1000.iloc[0] != b.test_net_per_1000.iloc[0]
    direct = evaluate_policy(y, t, top_k(scores["s_learner"], 0.5), value=10, cost=6)
    assert a.test_ci_low.iloc[0] == direct["ci_low"]
    assert a.test_coverage.iloc[0] == 0.5


def test_break_even_uses_contacted_denominator_and_actual_floor_coverage():
    _, _, _, candidates = fixture()
    bounds = break_even_table(candidates, value=10)
    targeted = bounds.loc[(bounds.policy == "s_learner") & (bounds.fraction == 0.5)].iloc[0]
    assert targeted.max_contact_cost == 10
    # A fractional budget can round down; divide by actual coverage, not requested reach.
    candidates = [
        {
            "policy": "s_learner",
            "fraction": 0.5,
            "coverage": 1 / 3,
            "incremental_conversions_per_1000": 1000 / 3,
        }
    ]
    assert break_even_table(candidates, 10).max_contact_cost.iloc[0] == pytest.approx(10)


@pytest.mark.parametrize(
    "values,costs",
    [([], [1]), ([1], []), ([-1], [1]), ([1], [float("nan")]), ([float("inf")], [1])],
)
def test_invalid_scenario_economics_are_rejected(values, costs):
    _, _, _, candidates = fixture()
    with pytest.raises(ValueError, match="economic axes"):
        select_scenarios(candidates, values, costs)


def test_negative_break_even_is_not_misrepresented_as_affordable_cost():
    candidates = [
        {"policy": "all", "fraction": 1, "coverage": 1, "incremental_conversions_per_1000": -100}
    ]
    table = break_even_table(candidates, 10)
    assert table.max_contact_cost.iloc[0] == -1
    assert not table.positive_uplift_on_validation.iloc[0]
