import numpy as np
import pandas as pd
import pytest

from src.x5 import FEATURES, RFMRanker, aggregate_purchases, assemble_features, validate_labels


def purchase_fixture():
    return pd.DataFrame(
        {
            "client_id": ["001", "001", "001", "002", "excluded"],
            "transaction_id": ["a", "a", "b", "c", "d"],
            "transaction_datetime": ["2020-01-01"] * 2 + ["2020-02-01"] * 3,
            "purchase_sum": [20, 20, 40, 30, 1000],
            "product_quantity": [1, 2, 1, 3, 10],
            "store_id": ["s1", "s1", "s2", "s1", "s3"],
            # Item amounts differ from repeated purchase_sum; receipt total is counted once.
            "trn_sum_from_iss": [999, 999, 999, 999, 999],
        }
    )


def test_receipt_aggregation_preserves_ids_and_does_not_count_item_lines_as_visits(tmp_path):
    path = tmp_path / "purchases.csv"
    purchase_fixture().to_csv(path, index=False)
    table, meta = aggregate_purchases(path, ["001", "002", "003"], tmp_path / "spill", "256MB")
    a = table.set_index("client_id").loc["001"]
    assert a.transactions == 2
    assert a.total_spend == 60
    assert a.mean_basket == 30
    assert a.recency_days == 1
    assert a.transactions_30d == 1
    assert a.spend_30d == 40
    assert a.mean_items == 2
    assert meta["receipt_count"] == 3
    assert meta["eligible_item_lines"] == 4
    assert set(table.client_id) == {"001", "002"}


@pytest.mark.parametrize(
    "column,value", [("transaction_datetime", "2020-01-02"), ("purchase_sum", 999)]
)
def test_receipt_conflict_is_rejected(tmp_path, column, value):
    rows = purchase_fixture()
    rows.loc[1, column] = value
    path = tmp_path / "purchases.csv"
    rows.to_csv(path, index=False)
    with pytest.raises(ValueError, match="invalid receipt"):
        aggregate_purchases(path, ["001"], tmp_path / "spill", "256MB")


def test_duplicate_labels_cannot_multiply_purchase_rows():
    labels = pd.DataFrame({"client_id": ["001", "001"], "treatment_flg": [0, 1], "target": [1, 0]})
    with pytest.raises(ValueError, match="nonunique"):
        validate_labels(labels)


def test_no_history_missing_age_and_leakage(tmp_path):
    path = tmp_path / "purchases.csv"
    purchase_fixture().to_csv(path, index=False)
    agg, _ = aggregate_purchases(path, ["001", "002", "003"], tmp_path / "spill", "256MB")
    labels = pd.DataFrame(
        {"client_id": ["001", "002", "003"], "treatment_flg": [0, 1, 0], "target": [0, 1, 1]}
    )
    clients = pd.DataFrame(
        {"client_id": ["001", "002", "003"], "age": [40, 150, np.nan], "gender": ["M", "F", None]}
    )
    a = assemble_features(labels, clients, agg)
    assert a.loc["003", "no_history"] == 1
    assert a.loc["003", "recency_days"] == 9999
    assert a.loc["002", "age"] == -1
    assert a.loc["003", "gender"] == "Unknown"
    labels["target"] = 1 - labels.target
    labels["treatment_flg"] = 1 - labels.treatment_flg
    b = assemble_features(labels, clients, agg)
    pd.testing.assert_frame_equal(a[FEATURES], b[FEATURES])


def test_rfm_uses_frozen_training_distribution():
    train = pd.DataFrame(
        {"recency_days": [1, 10, 20], "transactions": [10, 5, 1], "total_spend": [100, 50, 10]}
    )
    ranker = RFMRanker().fit(train)
    assert ranker.predict(train)[0] > ranker.predict(train)[2]
    before = ranker.predict(train.iloc[[0]])
    ranker.predict(
        pd.DataFrame({"recency_days": [0], "transactions": [10000], "total_spend": [1e9]})
    )
    np.testing.assert_array_equal(before, ranker.predict(train.iloc[[0]]))
