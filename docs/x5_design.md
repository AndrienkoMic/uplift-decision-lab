# X5 case: evidence and decisions

## Sources reviewed

- [Organizer data page](https://ods.ai/competitions/x5-retailhero-uplift-modeling/data):
  treatment marks communication, target marks subsequent purchase; purchases precede communication.
- [scikit-uplift dataset documentation](https://www.uplift-modeling.com/en/latest/api/datasets/fetch_x5.html):
  200,039 labeled customers, about 50% treatment, customer/purchase tables.
- [Library loader](https://github.com/maks-sh/scikit-uplift/blob/master/sklift/datasets/datasets.py):
  three public S3 files with published corruption-check MD5 values.

HTTP availability was checked before implementation. The downloader validates the published
checksums and records SHA-256 hashes. Dataset source terms apply separately from the code license.

## Causal status

Available descriptions do not provide the full randomization/assignment protocol. The binary
groups and approximately equal allocation do not prove randomization. This case therefore labels
all IPW policy values as conditional on assumed p=0.5 random assignment, no interference and
complete logging. It does not infer propensity from the observed group ratio or claim verified
causal profit. Acquire stronger assignment documentation or run a prospective randomized policy
test before treating the result as an operational recommendation.

No observational adjustment is silently substituted: a learned propensity score alone would
not establish absence of unmeasured confounding. Hillstrom remains the documented RCT case.

## Feature time boundary

Source purchases are described as pre-communication. The feature reference is the day after the
latest supplied receipt among labeled customers, with 30-day windows ending there. It is not a
claim about the true campaign date. Features use no outcomes/assignments and no post-treatment
columns. Customer age and gender use source descriptors; registration/redemption dates are not
model inputs. This random customer split evaluates a single supplied cohort, not future-time
generalization. Labels/outcomes from the contest's unlabeled test are never fabricated.

## Receipt semantics and leakage checks

CSV rows are item lines. Inspection of the raw extract shows purchase_sum is a repeated receipt
total. Aggregate by customer and transaction, take it once after verifying consistency, and sum
item quantities. Then form receipt frequency, basket and customer totals. Item-level
trn_sum_from_iss/trn_sum_from_red amounts are not needed here. Invalid dates, null required receipt
fields, inconsistent receipt amounts, timestamps or
stores, duplicate labels/client IDs and nonfinite features cause explicit failures. Returns are
not silently dropped. IDs are retained for joining/audit, excluded from prediction.

RFM percentile distributions and business segment thresholds fit only training features. Tests
verify that outcomes/assignment changes cannot alter features and that prediction does not
update the RFM reference. A hand-checkable receipt fixture protects against counting item lines
as purchases or accidentally summing repeated totals.

## Economics and evaluation

Default 100 RUB contribution and 2 RUB contact cost are scenario assumptions. No measured margin,
delivery rate, outcome-window duration or customer lifetime value is asserted. Policies compete
on validation estimated incremental net value, then one rule is locked. Predeclared diagnostic
20% comparisons and segment tables do not tune the policy on test. Approximate intervals are
conditional on fitted models/masks, omit training/top-K variation and are not multiplicity-adjusted.

The IPW benchmark can have substantial variance at a high baseline purchase rate. A future
cross-fitted augmented estimator may improve precision but does not repair unverified assignment.
