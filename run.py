"""Run the complete experiment-to-decision analysis: python run.py."""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import binary_experiment, load_data, split_data, validate_data
from src.economics import write_economics
from src.evaluation import get_qini, ranking_metrics
from src.experiment import ate, balance_table, randomization_check
from src.features import make_features
from src.models import fit_models, predict_scores
from src.policy import choose_policy, estimate, evaluate_policy, net_contributions, top_k

ROOT = Path(__file__).resolve().parent
FRACTIONS = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]


def make_mask(name, fraction, scores, n):
    if name == "none":
        return np.zeros(n, dtype=bool)
    if name == "all":
        return np.ones(n, dtype=bool)
    return top_k(np.zeros(n) if name == "random" else scores[name], fraction)


def figures(out, test, scores, policy_table, balance, currency="USD", channel="email"):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    palette = ["#0d9488", "#6366f1", "#f59e0b", "#e11d48"]
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    for (name, score), color in zip(scores.items(), palette):
        x, gain = get_qini(test.conversion, test.treatment, score)
        ax.plot(x, gain, label=name.replace("_", " "), color=color, lw=2)
    ax.plot([0, 1], [0, gain[-1]], "--", color="#94a3b8", label="random expectation")
    ax.set(
        xlabel="Share of eligible customers ranked",
        ylabel="Qini incremental conversions",
        title="Does ranking customers improve targeting? | untouched test",
    )
    ax.legend(frameon=False)
    fig.savefig(out / "figures/qini.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    table = policy_table.copy()
    selected_row = table.loc[table.label.str.startswith("selected:")].iloc[0]
    identical = (
        (table.coverage == selected_row.coverage)
        & (table.net_per_customer == selected_row.net_per_customer)
        & (table.ci_low == selected_row.ci_low)
        & (table.ci_high == selected_row.ci_high)
    )
    # Identical policies at 100% need one visual row, while the audit table retains every rule.
    table = table.loc[~identical | table.label.str.startswith("selected:")].reset_index(drop=True)
    table["label"] = table.label.str.replace("_", " ", regex=False)
    ax.errorbar(
        table.net_per_1000,
        np.arange(len(table)),
        xerr=np.array(
            [table.net_per_customer - table.ci_low, table.ci_high - table.net_per_customer]
        )
        * 1000,
        fmt="o",
        color="#0d9488",
        capsize=4,
    )
    ax.axvline(0, color="#94a3b8", ls="--")
    ax.set_yticks(np.arange(len(table)), table.label)
    ax.set(
        xlabel=f"Incremental net value per 1,000 eligible customers (assumed {currency})",
        title="Estimated policy value under assumed economics | 95% intervals",
    )
    fig.savefig(out / "figures/policy_value.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7), layout="constrained")
    table = balance.sort_values("smd")
    ax.barh(table.feature, table.smd, color="#6366f1")
    ax.axvline(-0.1, ls="--", color="#94a3b8")
    ax.axvline(0.1, ls="--", color="#94a3b8")
    ax.set(
        xlabel=f"Standardized mean difference: {channel} minus control",
        title="Pre-treatment balance | 0.1 is a heuristic, not a randomization proof",
    )
    fig.savefig(out / "figures/balance.png", dpi=180)
    plt.close(fig)


def markdown_table(frame, columns):
    # No optional tabulate dependency.
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        lines.append(
            "| "
            + " | ".join(
                f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c]) for c in columns
            )
            + " |"
        )
    return "\n".join(lines)


def write_report(out, summary, policies, metrics):
    primary = summary["ate_conversion"]
    selected = summary["selected_on_validation"]
    comparisons = summary["paired_test_comparisons"]
    policy_columns = [
        "label",
        "coverage",
        "incremental_conversions_per_1000",
        "net_per_1000",
        "ci_low",
        "ci_high",
    ]
    comparison_table = pd.DataFrame([{"comparator": k, **v} for k, v in comparisons.items()])
    comparison_columns = ["comparator", "net_per_customer", "ci_low", "ci_high"]
    deploy = all(row["ci_low"] > 0 for row in comparisons.values())
    if summary["randomization"]["srm_flag"]:
        decision = "Investigate the sample-ratio mismatch before making a causal recommendation."
    elif selected["policy"] == "none":
        decision = (
            "Validation selected no contact under the assumed economics. Keep that decision "
            "rather than selecting another rule from test diagnostics. Revisit economics and "
            "test a new policy prospectively if the business assumptions change."
        )
    elif selected["policy"] == "all" and comparisons["no contact"]["ci_low"] > 0:
        decision = (
            "Use contact-all as the candidate for a prospective campaign test under these "
            "economics. Validation selected this simple strategy and held-out net value versus "
            "no contact is positive with an approximate 95% interval above zero. ML targeting "
            "was not selected. Contact-all and random at 100% are the same rule, so their "
            "zero paired differences are identities, not failed superiority tests."
        )
    elif deploy:
        decision = (
            "The locked policy clears the exploratory comparison checks. Run a new prospective "
            "policy A/B test before deployment; the result is not a production guarantee."
        )
    else:
        decision = (
            "Do not claim demonstrated superiority of the selected policy. Its held-out evidence "
            "does not clear all comparisons with no contact, all, and random same coverage. "
            "Use a prospective policy A/B test to resolve the decision."
        )
    report = f"""# Uplift Decision Lab — experiment to decision

**Decision:** {decision}

## Business question
Should a retailer send the Mens email to everyone, no one, a random share, or customers ranked
by a model? Primary outcome: purchase within two weeks. Website visits are a secondary diagnostic.
The Womens arm is excluded from modeling, but retained in the original allocation check.

## Experiment evidence
- Original rows: {summary["quality"]["rows"]:,}; binary comparison: {summary["binary_rows"]:,}.
- Original 1:1:1 sample-ratio mismatch p-value: {summary["randomization"]["srm_p_value"]:.4f}.
- Missing required cells: {summary["quality"]["missing_cells"]}; duplicate profiles:
  {summary["quality"]["duplicate_profiles"]:,}. Identical profiles are not identified customers.
- Purchase ATE (all eligible binary rows, descriptive experiment estimate):
  {primary["ate"] * 100:.3f} percentage points,
  approximate 95% CI [{primary["ci_low"] * 100:.3f}, {primary["ci_high"] * 100:.3f}].
- Control purchase rate {primary["control_rate"]:.2%}; email {primary["treatment_rate"]:.2%}.
- Maximum absolute pre-treatment SMD: {summary["max_abs_smd"]:.3f}.
  SRM and balance checks can flag issues; neither proves correct assignment or absence of bias.

## Locked decision and economics
Train/validation/test sizes: {summary["split_sizes"]}. Treatment-only stratification; seed
{summary["config"]["seed"]}. Models see only training outcomes. Model and reach are chosen on
validation net value, including no-contact, contact-all and random alternatives. No test tuning.
Selected: **{selected["policy"]}**, requested coverage **{selected["fraction"]:.0%}**.

Assumed contribution per incremental purchase: **${summary["config"]["value"]:.2f}**;
contact cost: **${summary["config"]["cost"]:.2f}**. These are scenario inputs, not measured margin.
Break-even average conversion uplift among contacted users is cost/value (if value > 0).
No conversion probability is multiplied by observed post-treatment spend.

For each untouched test customer, use randomized assignment T, purchase Y, and policy pi(X):

`incremental_net_i = pi(X_i) * [value * (T_i Y_i / 0.5 - (1-T_i) Y_i / 0.5) - cost]`

The mean estimates incremental value versus no contact per **eligible customer**. Multiply
by 1,000 for a comparable audience. Conditional assignment probability is 0.5 because the
original three-arm experiment is filtered to Mens versus no email. It is not 1/3 or the observed
sample fraction. Top-K uses only pre-treatment scores and deterministic random tie-breaking.

## Held-out policy values
{markdown_table(policies, policy_columns)}

`ci_low` and `ci_high` above are net value per eligible customer. Intervals use the empirical
variance of IPW contributions and a normal approximation. Comparisons use paired differences
on the same customers, not overlap of separate confidence intervals.

Paired selected-minus-comparator results (net value per eligible customer):
{markdown_table(comparison_table, comparison_columns)}

![Policy value](figures/policy_value.png)

## Held-out ranking diagnostics
{markdown_table(metrics, ["model", "qini_auc_normalized", "auuc_normalized", "uplift_at_20pct"])}

Qini AUC and AUUC are **normalized scikit-uplift scores**, not dollar values or raw areas.
Uplift@20% is the treatment-minus-control purchase rate inside the top 20% (`overall` strategy).
These are descriptive diagnostics for all predeclared models; they do not reselect the policy.
The response baseline ranks by purchase probability learned on controls; it is not causal uplift.
Qini sorts tied scores as defined by sklift; the policy layer breaks ties explicitly.

![Qini](figures/qini.png)
![Balance](figures/balance.png)

## Limits and next experiment
- Purchases are rare. Model differences and normal intervals may be unstable; one split cannot
  establish general superiority. Confidence intervals are approximate, conditional on fitted
  models and fixed evaluated masks; they omit training variation and top-K threshold variation.
- Validation selection is optimistic there; test comparisons are exploratory, not multiplicity
  adjusted. Inspecting diagnostic test tables does not authorize trying another policy on test.
- The public extract cannot verify user identity, delivery, attrition, interference or logging
  completeness. Intent-to-treat assignment effects are not effects of actually opening an email.
- The dataset is an old retailer experiment. Customer mix, product offer, contact fatigue and
  opportunity costs may differ today. One campaign cannot estimate long-term customer value.
- A scalar conversion margin ignores basket heterogeneity. The spend ATE is reported separately
  in `summary.json`; revenue is not profit. Contact cost is charged to every intended recipient.
- Next: preregister the chosen policy versus a simple operational comparator, fix budget and
  decision rule, power the experiment using a business-relevant minimum effect, monitor delivery
  and guardrails, and measure net contribution prospectively.

## Reproduce and audit
Run `uv sync --locked` then `uv run python run.py`. Machine-readable outputs include
`summary.json`, `ranking_metrics.csv`, `policies_test.csv`, `policies_validation.csv`,
`balance.csv`, `sensitivity.csv`, local split IDs and local holdout predictions.
The manifest records data SHA-256, configuration, feature names and package versions.
Changing costs/values reruns a new scenario and validation selection; it is not an additional
independent test. Synthetic test fixtures are used only in unit tests, never in this report.

Sources: [Hillstrom original challenge](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html),
[dataset documentation](https://www.uplift-modeling.com/en/latest/api/datasets/fetch_hillstrom.html),
[scikit-uplift metrics](https://www.uplift-modeling.com/en/latest/api/metrics/index.html).
"""
    (out / "report.md").write_text(report, encoding="utf-8")
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="Offline original Hillstrom CSV (all 3 arms)")
    parser.add_argument("--output", type=Path, default=ROOT / "reports")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--value", type=float, default=50.0, help="Contribution per purchase")
    parser.add_argument("--cost", type=float, default=0.1, help="Cost per intended contact")
    args = parser.parse_args()
    if (
        args.iterations < 1
        or not np.isfinite([args.value, args.cost]).all()
        or min(args.value, args.cost) < 0
    ):
        parser.error("Positive iterations and finite nonnegative economics required")
    out = args.output
    (out / "figures").mkdir(parents=True, exist_ok=True)
    raw = load_data(ROOT / "data/raw", args.csv)
    quality = validate_data(raw)
    data = binary_experiment(raw)
    train, val, test = split_data(data, args.seed)
    features = {
        name: make_features(part)
        for name, part in [("train", train), ("validation", val), ("test", test)]
    }
    models = fit_models(
        features["train"], train.conversion, train.treatment, args.seed, args.iterations
    )
    val_scores = predict_scores(models, features["validation"])
    selected, candidates = choose_policy(
        val_scores, val.conversion, val.treatment, FRACTIONS, args.value, args.cost
    )
    # Selection is complete before test predictions or test outcomes are evaluated.
    scores = predict_scores(models, features["test"])
    masks = {
        "no contact": make_mask("none", 0, scores, len(test)),
        "contact all": make_mask("all", 1, scores, len(test)),
        "random same coverage": make_mask("random", selected["fraction"], scores, len(test)),
        f"selected: {selected['policy']}": make_mask(
            selected["policy"], selected["fraction"], scores, len(test)
        ),
    }
    masks["random at 20%"] = make_mask("random", 0.2, scores, len(test))
    # Additional predeclared 20% comparisons are diagnostics, not test selection candidates.
    for name in scores:
        masks[f"{name} at 20%"] = top_k(scores[name], 0.2)
    policies = pd.DataFrame(
        [
            {
                "label": label,
                **evaluate_policy(test.conversion, test.treatment, mask, args.value, args.cost),
            }
            for label, mask in masks.items()
        ]
    )
    selected_mask = masks[f"selected: {selected['policy']}"]
    selected_z = net_contributions(
        test.conversion, test.treatment, selected_mask, args.value, args.cost
    )
    comparisons = {
        label: estimate(
            selected_z
            - net_contributions(
                test.conversion, test.treatment, masks[label], args.value, args.cost
            )
        )
        for label in ["no contact", "contact all", "random same coverage"]
    }
    metrics = pd.DataFrame(
        [
            {"model": name, **values}
            for name, values in ranking_metrics(test.conversion, test.treatment, scores).items()
        ]
    )
    balance = balance_table(data)
    summary = {
        "quality": quality,
        "randomization": randomization_check(raw),
        "binary_rows": len(data),
        "max_abs_smd": float(balance.smd.abs().max()),
        "ate_conversion": ate(data.conversion, data.treatment),
        "ate_visit_secondary": ate(data.visit, data.treatment),
        "ate_spend_revenue": ate(data.spend, data.treatment),
        "split_sizes": {"train": len(train), "validation": len(val), "test": len(test)},
        "selected_on_validation": selected,
        "paired_test_comparisons": comparisons,
        "config": {
            "seed": args.seed,
            "iterations": args.iterations,
            "value": args.value,
            "cost": args.cost,
            "fractions": FRACTIONS,
            "propensity": 0.5,
        },
        "feature_names": list(features["train"].columns),
        "data_sha256": hashlib.sha256(raw.to_csv(index=False).encode()).hexdigest(),
        "versions": {
            name: version(name)
            for name in ["numpy", "pandas", "scipy", "scikit-learn", "scikit-uplift", "catboost"]
        },
    }
    # Sensitivity holds the selected policy fixed; it does not optimize using test outcomes.
    sensitivity = []
    for value in [20.0, 50.0, 100.0]:
        for cost in [0.01, 0.1, 0.5]:
            for label in [
                "no contact",
                "contact all",
                "random same coverage",
                f"selected: {selected['policy']}",
            ]:
                sensitivity.append(
                    {
                        "value": value,
                        "cost": cost,
                        "label": label,
                        **evaluate_policy(
                            test.conversion, test.treatment, masks[label], value, cost
                        ),
                    }
                )
    for name, table in [
        ("policies_test", policies),
        ("ranking_metrics", metrics),
        ("policies_validation", pd.DataFrame(candidates)),
        ("balance", balance),
        ("sensitivity", pd.DataFrame(sensitivity)),
    ]:
        table.to_csv(out / f"{name}.csv", index=False)
    pd.concat(
        [
            pd.DataFrame({"row_id": part.index, "split": name})
            for name, part in [("train", train), ("validation", val), ("test", test)]
        ]
    ).to_csv(out / "splits.csv", index=False)
    audit = test[["treatment", "conversion"]].copy()
    for name, score in scores.items():
        audit[name] = score
    audit["selected_contact"] = selected_mask.astype(int)
    audit.to_csv(out / "holdout_predictions.csv")
    figures(out, test, scores, policies, balance)
    decision = write_report(out, summary, policies, metrics)
    summary["decision"] = decision
    summary["economics"] = write_economics(
        out, candidates, scores, test.conversion, test.treatment, args.value, args.cost, "USD"
    )
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"Report: {out / 'report.md'}")
    print(f"Validation selected {selected['policy']} at {selected['fraction']:.0%} coverage.")
    print(decision)


if __name__ == "__main__":
    main()
