"""Stage 3 - Data preparation.

Cleans the validated tables (duplicates, missing values, formats) and produces a strictly
time-based train/test split plus an inner train/validation split used for hyper-parameter
selection. Every cleaning action is counted and written to ``reports/preparation_report.json``.
"""

from __future__ import annotations

import pandas as pd

from src.ingestion.ingest import GENRES
from src.utils import get_logger, load_config, path, write_json

log = get_logger("preparation")


def clean_users(df: pd.DataFrame, actions: dict) -> pd.DataFrame:
    n0 = len(df)
    df = df.drop(columns=["_raw_row"]).drop_duplicates(subset=["user_id", "age", "gender", "occupation", "zip_code"])
    actions["users_exact_duplicates_removed"] = n0 - len(df)
    n1 = len(df)
    df = df.sort_values("_ingested_at").drop_duplicates("user_id", keep="last")
    actions["users_conflicting_keys_resolved"] = n1 - len(df)

    df["occupation"] = df["occupation"].fillna("other").str.strip().str.lower()
    df["gender"] = df["gender"].where(df["gender"].isin(["M", "F"]), "U")
    actions["users_gender_set_unknown"] = int((df["gender"] == "U").sum())
    df["zip_code"] = df["zip_code"].fillna("UNKNOWN").str.strip().str.upper()

    df["age_imputed"] = df["age"].isna()
    occ_median = df.groupby("occupation")["age"].transform("median")
    df["age"] = df["age"].fillna(occ_median).fillna(df["age"].median()).round().astype(int)
    actions["users_age_imputed_by_occupation_median"] = int(df["age_imputed"].sum())
    df["age_bucket"] = pd.cut(df["age"], [0, 17, 24, 34, 44, 54, 120], labels=["<18", "18-24", "25-34", "35-44", "45-54", "55+"]).astype(str)
    return df.drop(columns=["_batch_id", "_ingested_at"]).sort_values("user_id").reset_index(drop=True)


def clean_items(df: pd.DataFrame, actions: dict) -> pd.DataFrame:
    n0 = len(df)
    df = df.drop(columns=["_raw_row"]).drop_duplicates(subset=[c for c in df.columns if not c.startswith("_")])
    actions["items_exact_duplicates_removed"] = n0 - len(df)
    df = df.drop_duplicates("item_id", keep="last")

    df["title"] = df["title"].str.strip().str.replace(r"\s+", " ", regex=True)
    title_year = pd.to_numeric(df["title"].str.extract(r"\((\d{4})\)\s*$")[0], errors="coerce")
    missing = df["release_date"].isna()
    df["release_date_imputed"] = missing
    df.loc[missing, "release_date"] = pd.to_datetime(title_year[missing].astype("Int64").astype(str) + "-01-01", errors="coerce")
    actions["items_release_date_imputed_from_title_year"] = int((missing & df["release_date"].notna()).sum())
    actions["items_release_date_still_missing"] = int(df["release_date"].isna().sum())
    df["release_year"] = df["release_date"].dt.year.astype("Int64")

    # video_release_date is empty for every record in the source data - dropped.
    df = df.drop(columns=["video_release_date", "_batch_id", "_ingested_at"])
    actions["items_columns_dropped"] = ["video_release_date (100% null)"]
    df["genres"] = df[GENRES].apply(lambda r: "|".join(g for g in GENRES if r[g] == 1), axis=1)
    df["n_genres"] = df[GENRES].sum(axis=1)
    return df.sort_values("item_id").reset_index(drop=True)


def clean_interactions(df: pd.DataFrame, actions: dict) -> pd.DataFrame:
    n0 = len(df)
    df = df.drop(columns=["_raw_row"]).drop_duplicates(subset=["user_id", "item_id", "rating", "timestamp"])
    actions["interactions_exact_duplicates_removed"] = n0 - len(df)
    n1 = len(df)
    # A user re-rating the same item: the most recent rating reflects the current preference.
    df = df.sort_values("timestamp").drop_duplicates(["user_id", "item_id"], keep="last")
    actions["interactions_conflicting_pairs_resolved_keep_latest"] = n1 - len(df)
    df = df.drop(columns=["_batch_id", "_ingested_at"])
    df["event_date"] = df["event_time"].dt.strftime("%Y-%m-%d")
    return df.sort_values(["timestamp", "user_id", "item_id"]).reset_index(drop=True)


def temporal_split(df: pd.DataFrame, fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Everything up to the (1 - fraction) timestamp quantile is history, everything after is future."""
    cutoff = int(df["timestamp"].quantile(1 - fraction))
    return df[df["timestamp"] <= cutoff], df[df["timestamp"] > cutoff], pd.Timestamp(cutoff, unit="s", tz="UTC")


def split_stats(train: pd.DataFrame, test: pd.DataFrame, cutoff: pd.Timestamp, threshold: int) -> dict:
    warm = set(test["user_id"]) & set(train["user_id"])
    return {
        "cutoff_utc": cutoff.isoformat(),
        "train_rows": len(train), "test_rows": len(test),
        "train_period": [train["event_time"].min().isoformat(), train["event_time"].max().isoformat()],
        "test_period": [test["event_time"].min().isoformat(), test["event_time"].max().isoformat()],
        "train_users": train["user_id"].nunique(), "test_users": test["user_id"].nunique(),
        "test_warm_users": len(warm), "test_cold_start_users": test["user_id"].nunique() - len(warm),
        "test_items_unseen_in_train": len(set(test["item_id"]) - set(train["item_id"])),
        "test_relevant_interactions": int((test["rating"] >= threshold).sum()),
    }


def run() -> dict:
    cfg = load_config()
    p = cfg["preparation"]
    actions: dict = {}
    users = clean_users(pd.read_parquet(path("validated", "users.parquet")), actions)
    items = clean_items(pd.read_parquet(path("validated", "items.parquet")), actions)
    inter = clean_interactions(pd.read_parquet(path("validated", "interactions.parquet")), actions)
    for k, v in actions.items():
        log.info("cleaning: %-52s %s", k, v)

    users.to_parquet(path("processed", "users.parquet"), index=False)
    items.to_parquet(path("processed", "items.parquet"), index=False)
    inter.to_parquet(path("processed", "interactions.parquet"), index=False)
    log.info("Clean tables -> users=%d items=%d interactions=%d", len(users), len(items), len(inter))

    train, test, cutoff = temporal_split(inter, p["test_fraction"])
    train_inner, val, val_cutoff = temporal_split(train, p["validation_fraction"])
    splits = path("processed", "splits")
    splits.mkdir(exist_ok=True)
    for name, df in {"train": train, "test": test, "train_inner": train_inner, "val": val}.items():
        df.to_parquet(splits / f"{name}.parquet", index=False)

    report = {
        "cleaning_actions": actions,
        "clean_rows": {"users": len(users), "items": len(items), "interactions": len(inter)},
        "split_strategy": f"global temporal cut-off at the {int(100 * (1 - p['test_fraction']))}th timestamp percentile",
        "test_split": split_stats(train, test, cutoff, p["relevance_threshold"]),
        "validation_split": split_stats(train_inner, val, val_cutoff, p["relevance_threshold"]),
    }
    assert train["timestamp"].max() < test["timestamp"].min(), "temporal leakage between train and test"
    write_json(report, path("reports", "preparation_report.json"))
    t = report["test_split"]
    log.info("Time-based split @ %s : train=%d (%s..%s) test=%d", cutoff.date(), t["train_rows"], t["train_period"][0][:10], t["train_period"][1][:10], t["test_rows"])
    log.info("Test users=%d (warm=%d, cold-start=%d); relevant test interactions=%d", t["test_users"], t["test_warm_users"], t["test_cold_start_users"], t["test_relevant_interactions"])
    log.info("Inner validation split @ %s : train_inner=%d val=%d", val_cutoff.date(), len(train_inner), len(val))
    log.info("No leakage check passed: max(train.ts) < min(test.ts)")
    return report


if __name__ == "__main__":
    run()
