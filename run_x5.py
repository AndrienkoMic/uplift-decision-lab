"""Second case study: contact policy selection with X5 purchase-history features."""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chisquare

from run import FRACTIONS, ROOT, figures, make_mask, markdown_table
from src.data import split_data
from src.evaluation import ranking_metrics
from src.experiment import ate, balance_table
from src.models import fit_models, predict_scores
from src.policy import choose_policy, estimate, evaluate_policy, net_contributions, top_k
from src.x5 import FEATURES, RFMRanker, load_x5


def write_report(out, summary, policies, metrics, segments):
    selected = summary["selected_on_validation"]
    effect = summary["purchase_difference"]
    comparison = pd.DataFrame(
        [{"comparator": k, **v} for k, v in summary["paired_test_comparisons"].items()]
    )
    policy_columns = [
        "label",
        "coverage",
        "incremental_conversions_per_1000",
        "net_per_1000",
        "ci_low",
        "ci_high",
    ]
    metric_columns = ["model", "qini_auc_normalized", "auuc_normalized", "uplift_at_20pct"]
    comparison_columns = ["comparator", "net_per_customer", "ci_low", "ci_high"]
    config = summary["config"]
    chosen = policies.loc[policies.label.str.startswith("selected:")].iloc[0]
    verdict = (
        "The holdout interval includes zero: positive net value is not established."
        if chosen.ci_low <= 0 <= chosen.ci_high
        else "Interpret this interval conditional on the assignment and economic assumptions."
    )
    report = f"""# X5 RetailHero — purchase propensity or incremental response?

**Locked candidate:** {selected["policy"]} at {selected["fraction"]:.0%} requested reach,
selected on validation net value. Evaluate this candidate in a new prospective randomized test
before an operational rollout. No guaranteed uplift-model advantage is assumed.

Held-out net value: **{chosen.net_per_1000:.2f} RUB per 1,000 eligible customers**;
approximate 95% CI [{chosen.ci_low * 1000:.2f}, {chosen.ci_high * 1000:.2f}]. {verdict}

**Evidence boundary:** the public sources describe treatment/control labels and approximately
50% treatment allocation, but do not supply a complete assignment protocol. Our policy estimator
assumes random assignment with p=0.5, no interference and complete outcome logging. Balance and
allocation diagnostics cannot validate those assumptions. The policy values below are conditional
benchmark estimates, not independently established causal profit or a deployment recommendation.

## Business question
Can contact targeting improve incremental purchases at a fixed contact budget compared with contact
all, random outreach, a control-response model and a familiar RFM customer ranking?
`target` is purchase after communication according to the source. The extract does not establish
the exact outcome-window duration, so this report does not reuse Hillstrom's two-week window.

Scenario inputs: **{config["value"]:.2f} RUB contribution per incremental purchase** and
**{config["cost"]:.2f} RUB per intended contact**. These are assumed economics,
not measured X5 margin.
The cost is charged to each intended recipient; we do not multiply uplift by past customer spend.

## Data and checks
- Labeled customers: {summary["quality"]["labeled_clients"]:,}; one row per customer after joins.
- Aggregated receipts: {summary["quality"]["receipt_count"]:,} in the labeled cohort.
- Provided purchase history: {summary["quality"]["history_start"]} to
  {summary["quality"]["history_end"]}.
- Feature reference: {summary["quality"]["feature_reference_date"]}; day after latest supplied
  pre-communication receipt. This is a feature anchor, not an invented campaign date.
- Customers with no history: {summary["quality"]["clients_without_history"]:,}; missing or
  implausible ages (outside 14–100): {summary["quality"]["ages_missing_or_outside_14_100"]:,}.
- Conditional 50:50 allocation check p-value: {summary["allocation"]["p_value"]:.4f}.
- Maximum absolute pre-treatment SMD: {summary["max_abs_smd"]:.3f}.
- Full-cohort purchase-rate difference: {effect["ate"] * 100:.3f} percentage points;
  approximate 95% CI [{effect["ci_low"] * 100:.3f}, {effect["ci_high"] * 100:.3f}].
- Control purchase rate {effect["control_rate"]:.2%}; contact group {effect["treatment_rate"]:.2%}.
  These are outcome rates, not incremental response rates.

## Features and baselines
Raw purchase rows are item lines. DuckDB first groups by `(client_id, transaction_id)` and takes
the repeated `purchase_sum` once, checking consistency. Item quantities are summed. Customer
features include recency, receipt frequency, total historical spend, mean basket, active days,
recent 30-day activity, items per receipt and store breadth, plus age and gender. Client ID,
treatment and the target never enter the model. Source clients' registration/redemption timestamps
and product-category features are excluded; no product-category table is loaded in this MVP.

RFM = equal-weight average of favorable recency, frequency and monetary percentiles relative to
**training customers only**. It is a high-value customer heuristic, not an uplift estimate.
The response baseline estimates purchase propensity on controls. CatBoost S/T learners retain
the same fixed complexity as Hillstrom. Categorical input is `gender`; missing numerics follow
explicit sentinel/zero rules with missingness flags, without fitted holdout imputation.

## Selection and held-out evaluation
60/20/20 split by customer, stratified only by assignment; seed {config["seed"]}.
Sizes: {summary["split_sizes"]}. Train fits models and RFM percentiles; validation chooses one
model/rule and coverage from the predeclared grid {FRACTIONS}; test never chooses another policy.

`net_i = pi(X_i) * [value * (T_i Y_i / 0.5 - (1-T_i) Y_i / 0.5) - cost]`

Values are relative to no contact per **eligible** customer, not per contacted customer.
Approximate 95% intervals use empirical contribution variance. Paired differences use the same
test customers; no conclusions are drawn from overlap of separate intervals.

{markdown_table(policies, policy_columns)}

`net_per_1000` is RUB per 1,000 eligible customers.
`ci_low`/`ci_high` are RUB per eligible customer.
Rows at 20% are predeclared diagnostics, not a second test-based policy selection.
At 100% reach all rankings select the same customers. The chart merges identical selected rows.

![Policy values](figures/policy_value.png)

Paired selected-minus-comparator estimates at the selected coverage:
{markdown_table(comparison, comparison_columns)}

## Ranking diagnostics
{markdown_table(metrics, metric_columns)}

Qini/AUUC are normalized scikit-uplift scores. Uplift@20% is the treatment-minus-control purchase
rate inside the top 20% (`overall` strategy). These metrics are not proof of economic benefit.
RFM and the response model are rankings, not estimates of a causal treatment effect.

![Qini](figures/qini.png)
![Balance](figures/balance.png)

## Business segments, fixed using training data
Segments use the training median of receipt count and training 75th percentile of recency.
“Lapsed” takes priority, then “frequent”, then “other”. They are heuristic labels, not a claim
that a customer has churned. Segment differences below are exploratory holdout diagnostics;
they do not alter the locked policy. Feature thresholds never use test outcomes.

{markdown_table(segments, ["segment", "n_control", "n_treatment", "ate", "ci_low", "ci_high"])}

## Limitations and next decision
- Assignment protocol is unverified in the available source documentation. A balanced treatment
  count is not evidence of exchangeability; observational targeting could confound these estimates.
- Costs, purchase contribution, delivery rate and opportunity costs are assumptions. Historical
  spend is not future margin. There is no post-treatment basket/margin outcome here.
- The normal intervals condition on fitted models and evaluated masks. They omit model-training
  and top-K threshold variation. Diagnostic comparisons and segments are not multiplicity-adjusted.
- High baseline purchase rates make IPW noisy. Frequent buyers may buy without contact.
  Strong predictive accuracy is not evidence of an incremental marketing opportunity.
- The source documents the purchase history as pre-communication. The extract alone cannot verify
  the true campaign cutoff, outcome window, identity completeness or delivery; no temporal
  generalization claim is made from this random customer split.
- Next: obtain/confirm assignment documentation, then prospectively test the locked candidate
  against the simplest operational alternative at a fixed budget with margin and contact guardrails.

## Reproduce and audit
`uv sync --locked` then `uv run python run_x5.py`. First run downloads about 647 MiB of compressed
data and aggregates it with a 2 GB DuckDB memory limit plus disk spill; cached runs reuse verified
customer aggregates. `--memory-limit` changes aggregation memory, not total Python process memory.
Raw data, customer features, split IDs and prediction exports stay local and are excluded from Git.
`summary.json` records source SHA-256 values, cache schema, features, configuration and versions.

Sources: [organizer data page](https://ods.ai/competitions/x5-retailhero-uplift-modeling/data),
[scikit-uplift dataset documentation](https://www.uplift-modeling.com/en/latest/api/datasets/fetch_x5.html),
[loader with source checksums](https://github.com/maks-sh/scikit-uplift/blob/master/sklift/datasets/datasets.py).
"""
    (out / "report.md").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "data/x5")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/x5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--value", type=float, default=100.0)
    parser.add_argument("--cost", type=float, default=2.0)
    parser.add_argument("--memory-limit", default="2GB")
    args = parser.parse_args()
    if (
        args.iterations < 1
        or not np.isfinite([args.value, args.cost]).all()
        or min(args.value, args.cost) < 0
    ):
        parser.error("Positive iterations and finite nonnegative economics required")
    out = args.output
    (out / "figures").mkdir(parents=True, exist_ok=True)
    data, quality = load_x5(args.cache, args.memory_limit)
    train, val, test = split_data(data, args.seed)
    models = fit_models(
        train[FEATURES],
        train.conversion,
        train.treatment,
        args.seed,
        args.iterations,
        categorical=["gender"],
    )
    rfm = RFMRanker().fit(train)
    val_scores = {**predict_scores(models, val[FEATURES]), "rfm_baseline": rfm.predict(val)}
    selected, candidates = choose_policy(
        val_scores, val.conversion, val.treatment, FRACTIONS, args.value, args.cost
    )
    scores = {**predict_scores(models, test[FEATURES]), "rfm_baseline": rfm.predict(test)}
    masks = {
        "no contact": make_mask("none", 0, scores, len(test)),
        "contact all": make_mask("all", 1, scores, len(test)),
        "random same coverage": make_mask("random", selected["fraction"], scores, len(test)),
        "RFM same coverage": make_mask("rfm_baseline", selected["fraction"], scores, len(test)),
        f"selected: {selected['policy']}": make_mask(
            selected["policy"], selected["fraction"], scores, len(test)
        ),
        "random at 20%": make_mask("random", 0.2, scores, len(test)),
    }
    masks.update({f"{name} at 20%": top_k(score, 0.2) for name, score in scores.items()})
    policies = pd.DataFrame(
        [
            {
                "label": name,
                **evaluate_policy(test.conversion, test.treatment, mask, args.value, args.cost),
            }
            for name, mask in masks.items()
        ]
    )
    selected_z = net_contributions(
        test.conversion,
        test.treatment,
        masks[f"selected: {selected['policy']}"],
        args.value,
        args.cost,
    )
    comparisons = {
        name: estimate(
            selected_z
            - net_contributions(test.conversion, test.treatment, masks[name], args.value, args.cost)
        )
        for name in ["no contact", "contact all", "random same coverage", "RFM same coverage"]
    }
    metrics = pd.DataFrame(
        [
            {"model": name, **values}
            for name, values in ranking_metrics(test.conversion, test.treatment, scores).items()
        ]
    )
    balance = balance_table(data, FEATURES)
    counts = data.treatment.value_counts().sort_index()
    _, allocation_p = chisquare(counts, np.repeat(len(data) / 2, 2))
    thresholds = {
        "recency_75pct": float(train.recency_days.quantile(0.75)),
        "frequency_median": float(train.transactions.median()),
    }
    segments = np.where(
        test.recency_days > thresholds["recency_75pct"],
        "lapsed",
        np.where(test.transactions > thresholds["frequency_median"], "frequent", "other"),
    )
    segment_rows = []
    for name in ["lapsed", "frequent", "other"]:
        part = test.loc[segments == name]
        segment_rows.append({"segment": name, **ate(part.conversion, part.treatment)})
    segments = pd.DataFrame(segment_rows)
    summary = {
        "dataset": "x5",
        "quality": quality,
        "assignment_status": "50:50 random assignment assumed; protocol unverified",
        "allocation": {"counts": counts.to_dict(), "p_value": float(allocation_p)},
        "max_abs_smd": float(balance.smd.abs().max()),
        "purchase_difference": ate(data.conversion, data.treatment),
        "selected_on_validation": selected,
        "paired_test_comparisons": comparisons,
        "split_sizes": {"train": len(train), "validation": len(val), "test": len(test)},
        "segment_thresholds_training_only": thresholds,
        "feature_names": FEATURES,
        "config": {
            "seed": args.seed,
            "iterations": args.iterations,
            "value": args.value,
            "cost": args.cost,
            "currency": "RUB",
            "propensity_assumption": 0.5,
            "fractions": FRACTIONS,
        },
        "features_sha256": hashlib.sha256(data[FEATURES].to_csv().encode()).hexdigest(),
        "versions": {
            n: version(n)
            for n in ["numpy", "pandas", "duckdb", "catboost", "scikit-uplift", "scikit-learn"]
        },
    }
    for name, table in [
        ("policies_test", policies),
        ("policies_validation", pd.DataFrame(candidates)),
        ("ranking_metrics", metrics),
        ("balance", balance),
        ("segments", segments),
    ]:
        table.to_csv(out / f"{name}.csv", index=False)
    pd.concat(
        [
            pd.DataFrame({"client_id": part.index, "split": name})
            for name, part in [("train", train), ("validation", val), ("test", test)]
        ]
    ).to_csv(out / "splits.csv", index=False)
    audit = test[["treatment", "conversion"]].copy()
    for name, score in scores.items():
        audit[name] = score
    audit["selected_contact"] = masks[f"selected: {selected['policy']}"]
    audit.to_csv(out / "holdout_predictions.csv")
    figures(out, test, scores, policies, balance, currency="RUB", channel="contact")
    write_report(out, summary, policies, metrics, segments)
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"X5 report: {out / 'report.md'}")
    print(f"Validation selected {selected['policy']} at {selected['fraction']:.0%} reach.")
    print("Policy values assume randomized assignment; source protocol is unverified.")


if __name__ == "__main__":
    main()
