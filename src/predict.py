"""Serving / batch prediction.

Generates top-K recommendations for one or more users with a trained model, reading user and item
features from the same feature-store snapshot that was used for training (train/serve consistency).

    python -m src.predict --user-ids 1 13 --k 10 --model hybrid
"""

from __future__ import annotations

import argparse
import pickle

import pandas as pd

from src.features.feature_store import FeatureStore
from src.utils import get_logger, path, read_json

log = get_logger("predict")


def recommend(user_ids: list[int], k: int = 10, model_name: str | None = None) -> pd.DataFrame:
    registry = read_json(path("models", "registry.json"))
    model_name = model_name or read_json(path("reports", "evaluation_results.json"))["best_model"]
    with open(path("models", f"{model_name}.pkl"), "rb") as fh:
        model = pickle.load(fh)
    fs = FeatureStore(registry["feature_snapshot"])
    titles = pd.read_parquet(path("processed", "items.parquet")).set_index("item_id")[["title", "genres"]]
    recs = model.recommend(user_ids, k)

    rows = []
    for u, items in recs.items():
        pair_f = fs.user_item_features([u] * len(items), items).set_index("item_id")
        item_f = fs.item_features(items)
        for rank, i in enumerate(items, 1):
            rows.append({"user_id": u, "rank": rank, "item_id": i, "title": titles.loc[i, "title"], "genres": titles.loc[i, "genres"],
                         "item_bayesian_avg": round(item_f.loc[i, "item_bayesian_avg"], 2),
                         "item_rating_count": int(item_f.loc[i, "item_rating_count"]),
                         "ui_genre_affinity": round(pair_f.loc[i, "ui_genre_affinity"], 3)})
    return pd.DataFrame(rows), model_name, fs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--user-ids", type=int, nargs="+", default=[1])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--model", default=None, help="model name (default: best model from evaluation)")
    args = parser.parse_args()

    df, model_name, fs = recommend(args.user_ids, args.k, args.model)
    profile = fs.user_features(args.user_ids)[["age", "gender", "occupation", "user_rating_count", "user_avg_rating", "user_preferred_genre", "is_cold_start"]]
    log.info("Model=%s  feature snapshot=%s (as_of %s)", model_name, fs.snapshot, fs.metadata["as_of"][:10])
    log.info("User profiles from the feature store:\n%s", profile.round(2).to_string())
    for u, grp in df.groupby("user_id", sort=False):
        log.info("Top-%d for user %d:\n%s", args.k, u, grp.drop(columns="user_id").to_string(index=False))


if __name__ == "__main__":
    main()
