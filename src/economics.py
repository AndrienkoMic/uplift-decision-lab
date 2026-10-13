"""Validation-selected economic scenarios; holdout outcomes never choose a rule."""

import hashlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from src.policy import evaluate_policy, top_k

GRIDS = {
    "USD": ([10, 20, 50, 100, 200], [0, 0.05, 0.1, 0.5, 1, 2]),
    "RUB": ([25, 50, 100, 200, 500], [0, 0.5, 1, 2, 5, 10]),
}
LABELS = {
    "none": "No contact",
    "all": "All",
    "random": "Random",
    "s_learner": "S",
    "t_learner": "T",
    "response_baseline": "Response",
    "rfm_baseline": "RFM",
}
COLORS = ["#e2e8f0", "#0d9488", "#94a3b8", "#6366f1", "#a855f7", "#f59e0b", "#e11d48"]


def select_scenarios(candidates, values, costs):
    """Only validation summaries enter this function; ties keep candidate order."""
    table = pd.DataFrame(candidates)
    columns = ["fraction", "coverage", "incremental_conversions_per_1000"]
    if table.empty or not {"policy", *columns} <= set(table):
        raise ValueError("Nonempty validation candidate table required")
    if not np.isfinite(table[columns].to_numpy(dtype=float)).all():
        raise ValueError("Finite validation candidate statistics required")
    if not table.fraction.between(0, 1).all() or not table.coverage.between(0, 1).all():
        raise ValueError("Candidate reach must be in [0,1]")
    values, costs = np.asarray(values, dtype=float), np.asarray(costs, dtype=float)
    for axis in [values, costs]:
        if axis.ndim != 1 or not len(axis) or not np.isfinite(axis).all() or (axis < 0).any():
            raise ValueError("Nonempty finite nonnegative economic axes required")
    rows = []
    for value in sorted(set(values)):
        for cost in sorted(set(costs)):
            net = value * table.incremental_conversions_per_1000 / 1000 - cost * table.coverage
            index = int(np.argmax(net.to_numpy()))
            chosen = table.iloc[index]
            rows.append(
                {
                    "value": value,
                    "cost": cost,
                    "policy": chosen.policy,
                    "fraction": chosen.fraction,
                    "validation_coverage": chosen.coverage,
                    "validation_net_per_1000": float(net.iloc[index] * 1000),
                }
            )
    return pd.DataFrame(rows)


def evaluate_scenarios(plans, scores, y, treatment):
    """Apply previously selected plans; never compare holdout policies to choose one."""
    rows = []
    masks = {}
    for row in plans.to_dict("records"):
        key = row["policy"], row["fraction"]
        if key not in masks:
            name, fraction = key
            if name in {"none", "all", "random"}:
                prediction = np.zeros(len(y))
            else:
                prediction = scores[name]
            masks[key] = top_k(prediction, fraction)
        result = evaluate_policy(y, treatment, masks[key], row["value"], row["cost"])
        rows.append({**row, **{f"test_{k}": v for k, v in result.items()}})
    return pd.DataFrame(rows)


def break_even_table(candidates, value):
    """Validation point estimate of max affordable cost per intended contact."""
    if not np.isfinite(value) or value < 0:
        raise ValueError("Finite nonnegative value required")
    table = pd.DataFrame(candidates).copy()
    table = table.loc[
        table.coverage > 0, ["policy", "fraction", "coverage", "incremental_conversions_per_1000"]
    ]
    table["value"] = value
    table["max_contact_cost"] = (
        value * table.incremental_conversions_per_1000 / 1000 / table.coverage
    )
    table["positive_uplift_on_validation"] = table.incremental_conversions_per_1000 > 0
    return table.reset_index(drop=True)


def plot_scenarios(out, table, currency):
    values, costs = sorted(table.value.unique()), sorted(table.cost.unique())
    lookup = table.set_index(["value", "cost"])
    names = list(LABELS)
    family = np.array([[names.index(lookup.loc[v, c].policy) for c in costs] for v in values])
    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    cmap = ListedColormap(COLORS)
    ax.imshow(
        family,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        norm=BoundaryNorm(np.arange(len(names) + 1) - 0.5, len(names)),
    )
    for i, value in enumerate(values):
        for j, cost in enumerate(costs):
            row = lookup.loc[value, cost]
            text = f"{LABELS[row.policy]}\n{row.fraction:.0%}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=10,
                color="#0f172a"
                if row.policy in {"none", "random", "response_baseline"}
                else "white",
            )
    ax.set(
        xticks=np.arange(len(costs)),
        xticklabels=[f"{c:g}" for c in costs],
        yticks=np.arange(len(values)),
        yticklabels=[f"{v:g}" for v in values],
        xlabel=f"Assumed cost per intended contact ({currency})",
        ylabel=f"Assumed contribution per incremental purchase ({currency})",
        title="Which rule and reach win on validation? | scenario assumptions",
    )
    used = [n for n in names if n in set(table.policy)]
    ax.legend(
        handles=[Patch(color=COLORS[names.index(n)], label=LABELS[n]) for n in used],
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        frameon=False,
    )
    fig.savefig(out / "figures/decision_map.png", dpi=180)
    plt.close(fig)

    net = np.array([[lookup.loc[v, c].test_net_per_1000 for c in costs] for v in values])
    limit = max(float(np.abs(net).max()), 1)
    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    im = ax.imshow(net, origin="lower", aspect="auto", cmap="RdBu", vmin=-limit, vmax=limit)
    for i, value in enumerate(values):
        for j, cost in enumerate(costs):
            row = lookup.loc[value, cost]
            symbol = "+" if row.test_ci_low > 0 else "−" if row.test_ci_high < 0 else "?"
            if row.policy == "none":
                symbol = "—"
            ax.text(
                j,
                i,
                f"{net[i, j]:,.0f}\n{symbol}",
                ha="center",
                va="center",
                fontsize=10,
                color="white" if abs(net[i, j]) > limit * 0.6 else "#0f172a",
            )
    ax.set(
        xticks=np.arange(len(costs)),
        xticklabels=[f"{c:g}" for c in costs],
        yticks=np.arange(len(values)),
        yticklabels=[f"{v:g}" for v in values],
        xlabel=f"Assumed cost per intended contact ({currency})",
        ylabel=f"Assumed contribution per incremental purchase ({currency})",
        title="Holdout net value of validation-selected rules | per 1,000 eligible customers",
    )
    fig.colorbar(im, ax=ax, label=f"Estimated incremental net value ({currency})")
    fig.text(
        0.5,
        -0.02,
        "+ / −: pointwise 95% interval above / below zero; ?: includes zero; "
        "—: no contact. Exploratory, conditional estimates.",
        ha="center",
        fontsize=9,
    )
    fig.savefig(out / "figures/scenario_value.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_economics(out, candidates, scores, y, treatment, value, cost, currency):
    """Generate exportable evidence from a fixed, documented scenario grid."""
    values, costs = GRIDS[currency]
    values, costs = sorted(set([*values, value])), sorted(set([*costs, cost]))
    plans = select_scenarios(candidates, values, costs)
    table = evaluate_scenarios(plans, scores, y, treatment)
    boundaries = break_even_table(candidates, value)
    table.to_csv(out / "economics_scenarios.csv", index=False)
    boundaries.to_csv(out / "break_even.csv", index=False)
    manifest = {
        "values": values,
        "costs": costs,
        "currency": currency,
        "selection": "validation IPW net-value point estimate; candidate-order ties",
        "test_role": "evaluate locked scenario plans, no reselection",
        "evidence_status": "Exploratory extension after original holdout inspection",
        "validation_candidates_sha256": hashlib.sha256(
            pd.DataFrame(candidates).to_csv(index=False).encode()
        ).hexdigest(),
        "grid_sha256": hashlib.sha256(json.dumps([values, costs]).encode()).hexdigest(),
    }
    (out / "economics_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    plot_scenarios(out, table, currency)
    default = table.loc[(table.value == value) & (table.cost == cost)].iloc[0]
    report = f"""# Economic decision map ({currency})

**Default scenario:** contribution {value:g}, cost {cost:g}; validation chooses
{default.policy} at {default.fraction:.0%} requested reach. Holdout net estimate:
{default.test_net_per_1000:.2f} {currency} per 1,000 eligible customers;
approximate 95% interval [{default.test_ci_low * 1000:.2f},
{default.test_ci_high * 1000:.2f}]. This does not establish superiority to another policy.

## When does the decision change?
Models fit train once. Each cell chooses the highest estimated net value on validation
from the same fixed policy/coverage candidates, then evaluates that locked rule on test.
The selection map uses no test outcomes. Costs apply to every intended contact;
contribution applies to an incremental purchase, not historical spend or baseline conversion.

![Validation decision map](figures/decision_map.png)

## What does the holdout support?
![Scenario value and uncertainty](figures/scenario_value.png)

Each cell shows net value per 1,000 **eligible** customers, including nonrecipients.
`+`/`−` indicate pointwise intervals wholly above/below zero; `?` includes zero;
`—` is the zero-valued no-contact rule. They are not multiplicity-adjusted deployment signals.
The full table is [economics_scenarios.csv](economics_scenarios.csv): confidence endpoints
`test_ci_low`/`test_ci_high` use {currency} per eligible customer (multiply by 1,000 for the chart).

## Break-even contact costs
For a fixed rule with positive reach:
`max_contact_cost = contribution × validation_incremental_purchases / validation_coverage`.
Below this boundary its validation point estimate is positive relative to no contact.
A zero/negative boundary provides no positive-value region at nonnegative costs.
The boundary is a point estimate, not a conservative spending limit or model prediction.
[break_even.csv](break_even.csv) exports every candidate at contribution {value:g} {currency}.

## Evidence boundary and reproducibility
- This grid was specified for an exploratory extension after the original holdout was inspected.
  It is not a new independent confirmatory test. Do not choose attractive economics or a new
  policy from the test heatmap and claim unbiased confirmation.
- All scenarios reuse the same customers and trained models. Their estimates are correlated;
  they are alternative assumptions, not separate experiments.
- Normal intervals condition on fitted models and policy masks. They omit training, threshold
  and validation-selection uncertainty and do not adjust for multiple comparisons.
- Assignment assumptions are inherited from the parent report: Hillstrom uses known p=0.5
  within the retained randomized arms; X5 assumes p=0.5 random assignment with an unverified
  full assignment protocol. A heatmap does not establish causality or future-time validity.
- Costs and purchase contribution are scenario assumptions. Confirm margin, delivery,
  assignment documentation and contact guardrails before a prospective policy test.
- Running the parent runner regenerates this report, both charts, tables and
  [manifest](economics_manifest.json), which records the grid and validation-candidate hashes.
"""
    (out / "economics_report.md").write_text(report, encoding="utf-8")
    parent = out / "report.md"
    if parent.exists():
        link = "[economic decision map](economics_report.md)"
        body = parent.read_text(encoding="utf-8")
        if link not in body:
            parent.write_text(
                body
                + "\n\n## Economic scenarios\n\nSee the "
                + link
                + " for validation-selected rules across assumed contact costs and purchase "
                + "contributions, holdout uncertainty and break-even costs. "
                + "This is an exploratory extension after original holdout inspection.\n",
                encoding="utf-8",
            )
    return manifest
