"""Verified X5 downloads and receipt-level features without loading 46M rows in pandas."""

import argparse
import hashlib
import json
from datetime import timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import requests

SOURCE = "https://sklift.s3.eu-west-2.amazonaws.com/"
FILES = {
    "uplift_train.csv.gz": "2720bbb659daa9e0989b2777b6a42d19",
    "clients.csv.gz": "b9cdeb2806b732771de03e819b3354c5",
    "purchases.csv.gz": "48d2de13428e24e8b61d66fef02957a8",
}
# Source MD5 is a corruption check, not a security guarantee. SHA-256 is recorded too.
NUMERIC = [
    "age",
    "transactions",
    "total_spend",
    "mean_basket",
    "recency_days",
    "active_days",
    "history_days",
    "transactions_30d",
    "spend_30d",
    "spend_share_30d",
    "mean_items",
    "store_count",
    "purchase_frequency",
    "no_history",
    "age_missing",
]
FEATURES = NUMERIC + ["gender"]


def file_hash(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(cache):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, expected in FILES.items():
        path = cache / name
        if not path.exists():
            partial = cache / (name + ".part")
            print(f"Downloading {name}", flush=True)
            with requests.get(SOURCE + name, stream=True, timeout=(15, 120)) as response:
                response.raise_for_status()
                count = 0
                with partial.open("wb") as stream:
                    for chunk in response.iter_content(2**20):
                        stream.write(chunk)
                        count += len(chunk)
                        if count // (100 * 2**20) != (count - len(chunk)) // (100 * 2**20):
                            print(f"  {count / 2**20:.0f} MiB received", flush=True)
            if file_hash(partial, "md5") != expected:
                raise ValueError(f"Checksum mismatch: {partial}; original cache preserved")
            partial.replace(path)
        if file_hash(path, "md5") != expected:
            raise ValueError(f"Checksum mismatch: {path}; inspect or replace this cache file")
        manifest[name] = {
            "url": SOURCE + name,
            "sha256": file_hash(path),
            "bytes": path.stat().st_size,
        }
    return manifest


def validate_labels(labels):
    required = {"client_id", "treatment_flg", "target"}
    if not required <= set(labels):
        raise ValueError("X5 label schema is incomplete")
    if labels[list(required)].isna().any().any() or labels.client_id.duplicated().any():
        raise ValueError("Missing label values or nonunique customers")
    if set(labels.treatment_flg.unique()) != {0, 1} or not labels.target.isin([0, 1]).all():
        raise ValueError("Binary treatment with both arms and binary target required")


def aggregate_purchases(path, client_ids, work_dir, memory_limit="2GB"):
    """One receipt per customer/transaction; purchase_sum is a repeated receipt total."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        con.execute("SET memory_limit = ?", [memory_limit])
        con.execute("SET temp_directory = ?", [str(work_dir.resolve())])
        con.execute("SET threads = 2")
        con.register("eligible", pd.DataFrame({"client_id": client_ids}))
        con.execute(
            """
            CREATE TEMP TABLE receipts AS
            SELECT p.client_id, p.transaction_id,
                   min(cast(transaction_datetime AS TIMESTAMP)) AS purchased_at,
                   min(cast(purchase_sum AS DOUBLE)) AS basket_sum,
                   sum(cast(product_quantity AS DOUBLE)) AS items,
                   min(store_id) AS store_id,
                   count(*) AS item_lines,
                   count(*) FILTER (WHERE transaction_id IS NULL
                        OR transaction_datetime IS NULL OR purchase_sum IS NULL
                        OR product_quantity IS NULL OR store_id IS NULL) AS invalid_lines,
                   count(DISTINCT transaction_datetime) AS date_variants,
                   count(DISTINCT store_id) AS store_variants,
                   count(DISTINCT cast(purchase_sum AS DOUBLE)) AS amount_variants
            FROM read_csv(?, header=true, all_varchar=true) p
            INNER JOIN eligible e ON p.client_id=e.client_id
            GROUP BY p.client_id, p.transaction_id
        """,
            [str(Path(path).resolve())],
        )
        quality = con.execute("""
            SELECT count(*), min(purchased_at), max(purchased_at),
                   count(*) FILTER (WHERE basket_sum IS NULL OR purchased_at IS NULL
                        OR invalid_lines > 0 OR date_variants != 1 OR store_variants != 1
                        OR amount_variants != 1),
                   sum(item_lines)
            FROM receipts
        """).fetchone()
        if not quality[0] or quality[3]:
            raise ValueError("Empty purchase history or invalid receipt dates/amounts")
        # The source documents history before communication. This is a feature reference,
        # not an invented campaign date or a claim about the outcome window's length.
        reference = pd.Timestamp(quality[2].date() + timedelta(days=1))
        aggregates = con.execute(
            """
            SELECT client_id, count(*) AS transactions, sum(basket_sum) AS total_spend,
                   avg(basket_sum) AS mean_basket, avg(items) AS mean_items,
                   count(DISTINCT store_id) AS store_count,
                   count(DISTINCT cast(purchased_at AS DATE)) AS active_days,
                   date_diff('day', max(purchased_at), cast(? AS TIMESTAMP)) AS recency_days,
                   date_diff('day', min(purchased_at), cast(? AS TIMESTAMP)) AS history_days,
                   count(*) FILTER (WHERE purchased_at >= cast(? AS TIMESTAMP)
                        - INTERVAL 30 DAY) AS transactions_30d,
                   coalesce(sum(basket_sum) FILTER (WHERE purchased_at >= cast(? AS TIMESTAMP)
                        - INTERVAL 30 DAY), 0) AS spend_30d
            FROM receipts GROUP BY client_id ORDER BY client_id
        """,
            [reference.to_pydatetime()] * 4,
        ).df()
        metadata = {
            "receipt_count": quality[0],
            "history_start": str(quality[1]),
            "eligible_item_lines": int(quality[4]),
            "history_end": str(quality[2]),
            "feature_reference_date": str(reference),
            "reference_note": "Day after latest provided pre-communication receipt; "
            "not the campaign date or an observed outcome-window boundary.",
        }
        return aggregates, metadata
    finally:
        con.close()


def assemble_features(labels, clients, aggregates):
    validate_labels(labels)
    if clients.client_id.isna().any() or clients.client_id.duplicated().any():
        raise ValueError("Client table requires unique nonmissing IDs")
    if aggregates.client_id.duplicated().any():
        raise ValueError("Aggregates must contain one row per client")
    frame = labels.merge(
        clients[["client_id", "age", "gender"]],
        on="client_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if (frame._merge != "both").any():
        raise ValueError("Some labeled clients are absent from client table")
    frame = frame.drop(columns="_merge").merge(
        aggregates, on="client_id", how="left", validate="one_to_one"
    )
    frame["no_history"] = frame.transactions.isna().astype(int)
    frame["age_missing"] = (~frame.age.between(14, 100)).astype(int)
    frame["age"] = frame.age.where(frame.age.between(14, 100), -1)
    frame["gender"] = frame.gender.fillna("Unknown").astype(str)
    frame["recency_days"] = frame.recency_days.fillna(9999)
    history_columns = [c for c in aggregates if c != "client_id"]
    frame[history_columns] = frame[history_columns].fillna(0)
    frame["spend_share_30d"] = np.where(
        frame.total_spend > 0, frame.spend_30d / frame.total_spend.clip(lower=1e-9), 0
    )
    frame["purchase_frequency"] = frame.transactions / frame.history_days.clip(lower=1)
    frame = frame.rename(columns={"target": "conversion", "treatment_flg": "treatment"})
    frame = frame.set_index("client_id").sort_index()
    if not np.isfinite(frame[NUMERIC].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite customer features")
    return frame


def load_x5(cache, memory_limit="2GB"):
    cache = Path(cache)
    hashes = download(cache)
    labels = pd.read_csv(cache / "uplift_train.csv.gz", dtype={"client_id": str})
    clients = pd.read_csv(cache / "clients.csv.gz", dtype={"client_id": str})
    validate_labels(labels)
    aggregate_cache = cache / "customer_aggregates.parquet"
    metadata_cache = cache / "aggregates.json"
    cache_key = {"schema_version": 2, "sources": hashes}
    metadata = json.loads(metadata_cache.read_text()) if metadata_cache.exists() else {}
    if aggregate_cache.exists() and metadata.get("cache_key") == cache_key:
        aggregates = duckdb.sql(
            "SELECT * FROM read_parquet(?)", params=[str(aggregate_cache.resolve())]
        ).df()
    else:
        print("Aggregating item rows into receipts and customer features", flush=True)
        aggregates, metadata = aggregate_purchases(
            cache / "purchases.csv.gz", labels.client_id, cache / "spill", memory_limit
        )
        con = duckdb.connect()
        try:
            con.register("aggregates", aggregates)
            con.execute("COPY aggregates TO ? (FORMAT PARQUET)", [str(aggregate_cache.resolve())])
        finally:
            con.close()
        metadata["cache_key"] = cache_key
        metadata_cache.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    frame = assemble_features(labels, clients, aggregates)
    con = duckdb.connect()
    try:
        con.register("features", frame[FEATURES].reset_index())
        con.execute(
            "COPY features TO ? (FORMAT PARQUET)",
            [str((cache / "customer_features.parquet").resolve())],
        )
    finally:
        con.close()
    return frame, {
        **metadata,
        "source_hashes": hashes,
        "labeled_clients": len(frame),
        "clients_without_history": int(frame.no_history.sum()),
        "ages_missing_or_outside_14_100": int(frame.age_missing.sum()),
        "missing_gender": int((frame.gender == "Unknown").sum()),
    }


class RFMRanker:
    """Equal-weight R/F/M percentiles relative to TRAIN customers, no outcomes used."""

    columns = ["recency_days", "transactions", "total_spend"]

    def fit(self, frame):
        self.distributions = [np.sort(frame[c].to_numpy()) for c in self.columns]
        return self

    def predict(self, frame):
        ranks = [
            np.searchsorted(dist, frame[c], side="right") / len(dist)
            for c, dist in zip(self.columns, self.distributions)
        ]
        return ((1 - ranks[0]) + ranks[1] + ranks[2]) / 3


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/x5"))
    args = parser.parse_args()
    print(json.dumps(download(args.cache), indent=2))
