"""Stage 4 - Data transformation.

Builds derived datasets from the *training* period only (so nothing from the test period leaks
into anything a model can see):

* ``user_activity_summary``  - per-user volume, rating behaviour and activity window
* ``item_popularity``        - per-item volume, mean / Bayesian-average rating, popularity rank
* ``item_genres``            - long (item, genre) table for genre-level analytics
* ``user_item_matrix``       - sparse CSR rating matrix + id <-> index mappings
* ``daily_activity``         - interactions per day (data-drift / volume monitoring)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.ingestion.ingest import GENRES
from src.utils import get_logger, load_config, path, write_json

log = get_logger("transformation")


def user_activity_summary(train: pd.DataFrame, threshold: int) -> pd.DataFrame:
    g = train.groupby("user_id")
    out = g.agg(
        n_interactions=("item_id", "size"),
        mean_rating=("rating", "mean"),
        std_rating=("rating", "std"),
        n_positive=("rating", lambda r: int((r >= threshold).sum())),
        first_event=("event_time", "min"),
        last_event=("event_time", "max"),
        active_days=("event_date", "nunique"),
    ).reset_index()
    out["std_rating"] = out["std_rating"].fillna(0.0)
    out["positive_ratio"] = out["n_positive"] / out["n_interactions"]
    out["tenure_days"] = (out["last_event"] - out["first_event"]).dt.total_seconds() / 86400
    return out


def item_popularity(train: pd.DataFrame, items: pd.DataFrame, prior_m: int) -> pd.DataFrame:
    mu = train["rating"].mean()
    agg = train.groupby("item_id").agg(n_ratings=("rating", "size"), mean_rating=("rating", "mean"), n_users=("user_id", "nunique")).reset_index()
    out = items[["item_id", "title", "genres"]].merge(agg, on="item_id", how="left")
    out[["n_ratings", "n_users"]] = out[["n_ratings", "n_users"]].fillna(0).astype(int)
    # Bayesian average: shrinks items with few ratings toward the global mean.
    out["bayesian_avg"] = (out["n_ratings"] * out["mean_rating"].fillna(mu) + prior_m * mu) / (out["n_ratings"] + prior_m)
    out["popularity_rank"] = out["n_ratings"].rank(ascending=False, method="min").astype(int)
    out["popularity_pct"] = out["n_ratings"] / max(len(train), 1)
    return out.sort_values("popularity_rank").reset_index(drop=True)


def item_genres_long(items: pd.DataFrame) -> pd.DataFrame:
    long = items.melt(id_vars=["item_id"], value_vars=GENRES, var_name="genre", value_name="flag")
    return long[long["flag"] == 1].drop(columns="flag").sort_values(["item_id", "genre"]).reset_index(drop=True)


def user_item_matrix(train: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame) -> tuple[sp.csr_matrix, dict]:
    user_ids, item_ids = np.sort(users["user_id"].unique()), np.sort(items["item_id"].unique())
    u_idx = {int(u): i for i, u in enumerate(user_ids)}
    i_idx = {int(m): j for j, m in enumerate(item_ids)}
    rows = train["user_id"].map(u_idx).to_numpy()
    cols = train["item_id"].map(i_idx).to_numpy()
    mat = sp.csr_matrix((train["rating"].to_numpy(dtype=np.float32), (rows, cols)), shape=(len(user_ids), len(item_ids)))
    return mat, {"user_ids": user_ids.tolist(), "item_ids": item_ids.tolist()}


def run() -> dict:
    cfg = load_config()
    threshold = cfg["preparation"]["relevance_threshold"]
    users = pd.read_parquet(path("processed", "users.parquet"))
    items = pd.read_parquet(path("processed", "items.parquet"))
    train = pd.read_parquet(path("processed", "splits", "train.parquet"))
    out = path("derived")

    activity = user_activity_summary(train, threshold)
    popularity = item_popularity(train, items, cfg["features"]["bayesian_prior_count"])
    genres = item_genres_long(items)
    daily = train.groupby("event_date").agg(n_interactions=("rating", "size"), active_users=("user_id", "nunique"), mean_rating=("rating", "mean")).reset_index()
    matrix, index = user_item_matrix(train, users, items)

    activity.to_parquet(out / "user_activity_summary.parquet", index=False)
    popularity.to_parquet(out / "item_popularity.parquet", index=False)
    genres.to_parquet(out / "item_genres.parquet", index=False)
    daily.to_parquet(out / "daily_activity.parquet", index=False)
    sp.save_npz(out / "user_item_matrix.npz", matrix)
    write_json(index, out / "user_item_matrix_index.json")

    density = matrix.nnz / (matrix.shape[0] * matrix.shape[1])
    summary = {
        "source": "train split only (leak-free)",
        "user_activity_summary": {"rows": len(activity), "median_interactions": float(activity["n_interactions"].median())},
        "item_popularity": {"rows": len(popularity), "items_with_ratings": int((popularity["n_ratings"] > 0).sum()),
                            "top5": popularity.head(5)[["title", "n_ratings", "bayesian_avg"]].to_dict("records")},
        "item_genres": {"rows": len(genres)},
        "daily_activity": {"rows": len(daily)},
        "user_item_matrix": {"shape": list(matrix.shape), "nnz": int(matrix.nnz), "density_pct": round(100 * density, 3),
                             "sparsity_pct": round(100 * (1 - density), 3)},
    }
    write_json(summary, path("reports", "transformation_summary.json"))
    log.info("user_activity_summary rows=%d (median interactions/user=%.0f)", len(activity), activity["n_interactions"].median())
    log.info("item_popularity rows=%d; top-3: %s", len(popularity), ", ".join(popularity["title"].head(3)))
    log.info("item_genres (long) rows=%d; daily_activity rows=%d", len(genres), len(daily))
    log.info("user_item_matrix shape=%s nnz=%d sparsity=%.2f%%", matrix.shape, matrix.nnz, 100 * (1 - density))
    return summary


if __name__ == "__main__":
    run()
