import numpy as np
import pandas as pd

from src.features.feature_store import compute_item_features, compute_user_features
from src.ingestion.ingest import GENRES
from src.preparation.prepare import clean_interactions, temporal_split
from src.utils import load_config


def _interactions():
    ts = np.arange(1, 11) * 86400 + 880000000
    df = pd.DataFrame({"user_id": [1, 1, 2, 2, 3, 3, 1, 2, 3, 1], "item_id": [1, 2, 1, 3, 2, 3, 3, 2, 1, 4],
                       "rating": [5, 3, 4, 2, 5, 4, 1, 3, 4, 5], "timestamp": ts})
    df["event_time"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["event_date"] = df["event_time"].dt.strftime("%Y-%m-%d")
    return df


def _items():
    items = pd.DataFrame({"item_id": [1, 2, 3, 4], "release_date": pd.to_datetime(["1995-01-01"] * 4), "n_genres": 1})
    for g in GENRES:
        items[g] = 0
    items.loc[[0, 1], "Drama"] = 1
    items.loc[[2, 3], "Comedy"] = 1
    items["release_year"] = 1995
    return items


def test_temporal_split_has_no_leakage():
    train, test, cutoff = temporal_split(_interactions(), 0.2)
    assert train["timestamp"].max() < test["timestamp"].min()
    assert len(train) == 8 and len(test) == 2
    assert (train["event_time"] <= cutoff).all()


def test_conflicting_pairs_keep_latest():
    df = _interactions()
    dup = df.iloc[[0]].copy()
    dup["rating"], dup["timestamp"] = 1, dup["timestamp"] + 10
    dup["_raw_row"], dup["_batch_id"], dup["_ingested_at"] = 0, "b", "t"
    df["_raw_row"], df["_batch_id"], df["_ingested_at"] = range(len(df)), "b", "t"
    out = clean_interactions(pd.concat([df, dup]), {})
    assert out.query("user_id == 1 and item_id == 1")["rating"].tolist() == [1]


def test_features_are_point_in_time():
    cfg = load_config()
    inter, items = _interactions(), _items()
    users = pd.DataFrame({"user_id": [1, 2, 3, 4], "age": [20, 30, 40, 50], "age_bucket": ["18-24", "25-34", "35-44", "45-54"],
                          "gender": ["M", "F", "M", "F"], "occupation": ["student"] * 4})
    train, _, cutoff = temporal_split(inter, 0.2)
    item_f = compute_item_features(train, items, cutoff, cfg)
    user_f = compute_user_features(train, users, items, item_f, cutoff, cfg).set_index("user_id")
    assert user_f.loc[1, "user_rating_count"] == (train["user_id"] == 1).sum()   # future ratings excluded
    assert bool(user_f.loc[4, "is_cold_start"]) and user_f.loc[4, "user_rating_count"] == 0
    assert (user_f["user_days_since_last_interaction"].dropna() >= 0).all()
    assert item_f.set_index("item_id").loc[4, "item_rating_count"] == 0          # item 4 only rated in the test period
