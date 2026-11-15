"""Stage 5 - Feature management (custom offline feature store).

Features are materialised as *point-in-time snapshots*: a snapshot is computed only from
interactions at or before its ``as_of`` timestamp, so training never sees the future.

* ``validation`` snapshot - built from ``train_inner`` (used for hyper-parameter selection)
* ``train`` snapshot      - built from the full training period (used for final training,
                            evaluation and serving / ``predict.py``)

The same :class:`FeatureStore` API is used by the training code and by the prediction script,
which guarantees train/serve consistency.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.features.registry import ITEM_FEATURES, USER_FEATURES, USER_ITEM_FEATURES, catalog_rows
from src.ingestion.ingest import GENRES
from src.utils import get_logger, load_config, path, read_json, write_json

log = get_logger("features")
GENRE_SHRINKAGE = 5.0


def _days(delta: pd.Series) -> pd.Series:
    return delta.dt.total_seconds() / 86400


def compute_item_features(inter: pd.DataFrame, items: pd.DataFrame, as_of: pd.Timestamp, cfg: dict) -> pd.DataFrame:
    m = cfg["features"]["bayesian_prior_count"]
    mu = inter["rating"].mean()
    agg = inter.groupby("item_id").agg(
        item_rating_count=("rating", "size"), item_avg_rating=("rating", "mean"),
        item_rating_std=("rating", "std"), last=("event_time", "max"),
    )
    f = items.set_index("item_id")[GENRES + ["release_date", "release_year", "n_genres"]].join(agg, how="left")
    f["item_rating_count"] = f["item_rating_count"].fillna(0).astype(int)
    f["item_bayesian_avg"] = (f["item_rating_count"] * f["item_avg_rating"].fillna(mu) + m * mu) / (f["item_rating_count"] + m)
    f["item_rating_std"] = f["item_rating_std"].fillna(0.0)
    f["item_popularity_rank"] = f["item_rating_count"].rank(ascending=False, method="min").astype(int)
    f["item_log_popularity"] = np.log1p(f["item_rating_count"])
    f["item_days_since_last_rating"] = _days(as_of - f["last"])
    f["item_release_year"] = f["release_year"].astype("Int64")
    f["item_age_years"] = _days(as_of - f["release_date"].dt.tz_localize("UTC")) / 365.25
    f["item_primary_genre"] = f[GENRES].idxmax(axis=1)
    f["item_n_genres"] = f["n_genres"].astype(int)
    f = f.rename(columns={g: f"genre_{g}" for g in GENRES})
    cols = [fd.name for fd in ITEM_FEATURES]
    return f[cols].reset_index()


def compute_user_genre_profile(inter: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """Shrunk mean rating deviation per (user, genre): sum(r - mu_u) / (n_ug + lambda)."""
    x = inter[["user_id", "item_id", "rating"]].copy()
    x["dev"] = x["rating"] - x.groupby("user_id")["rating"].transform("mean")
    g = items.set_index("item_id")[GENRES]
    flags = g.loc[x["item_id"]].to_numpy()
    dev_sum = pd.DataFrame(flags * x["dev"].to_numpy()[:, None], columns=GENRES).groupby(x["user_id"].to_numpy()).sum()
    cnt = pd.DataFrame(flags, columns=GENRES).groupby(x["user_id"].to_numpy()).sum()
    profile = dev_sum / (cnt + GENRE_SHRINKAGE)
    profile.index.name = "user_id"
    return profile


def compute_user_features(inter: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame, item_feats: pd.DataFrame,
                          as_of: pd.Timestamp, cfg: dict) -> pd.DataFrame:
    thr = cfg["preparation"]["relevance_threshold"]
    mu = inter["rating"].mean()
    agg = inter.groupby("user_id").agg(
        user_rating_count=("rating", "size"), user_avg_rating=("rating", "mean"), user_rating_std=("rating", "std"),
        user_positive_ratio=("rating", lambda r: (r >= thr).mean()), user_active_days=("event_date", "nunique"),
        first=("event_time", "min"), last=("event_time", "max"),
    )
    agg["user_tenure_days"] = _days(agg["last"] - agg["first"])
    agg["user_days_since_last_interaction"] = _days(as_of - agg["last"])

    enriched = inter.merge(item_feats[["item_id", "item_log_popularity", "item_release_year"]], on="item_id")
    agg = agg.join(enriched.groupby("user_id").agg(
        user_mainstreamness=("item_log_popularity", "mean"),
        user_avg_release_year=("item_release_year", lambda s: s.astype(float).mean()),
    ))

    genre_long = inter[["user_id", "item_id", "rating"]].merge(
        items.melt(id_vars="item_id", value_vars=GENRES, var_name="genre", value_name="flag").query("flag == 1"), on="item_id")
    counts = genre_long.groupby(["user_id", "genre"]).size().unstack(fill_value=0)
    p = counts.div(counts.sum(axis=1), axis=0)
    agg["user_genre_entropy"] = -(p * np.log2(p.where(p > 0, 1))).sum(axis=1)
    pos = genre_long[genre_long["rating"] >= thr].groupby(["user_id", "genre"]).size().unstack(fill_value=0)
    pos = pos.where(counts.reindex_like(pos) >= cfg["features"]["min_genre_ratings"], 0)
    agg["user_preferred_genre"] = pos.idxmax(axis=1).where(pos.max(axis=1) > 0)

    f = users.set_index("user_id")[["age", "age_bucket", "gender", "occupation"]].join(agg, how="left")
    f["is_cold_start"] = f["user_rating_count"].isna()
    defaults = {"user_rating_count": 0, "user_avg_rating": mu, "user_rating_std": 0.0, "user_positive_ratio": 0.0,
                "user_active_days": 0, "user_tenure_days": 0.0, "user_genre_entropy": 0.0,
                "user_mainstreamness": item_feats["item_log_popularity"].mean(),
                "user_avg_release_year": float(item_feats["item_release_year"].astype(float).mean()),
                "user_preferred_genre": "none"}
    f = f.fillna(defaults)
    f[["user_rating_count", "user_active_days"]] = f[["user_rating_count", "user_active_days"]].astype(int)
    return f[[fd.name for fd in USER_FEATURES]].reset_index()


class FeatureStore:
    """Read/write access to a materialised feature snapshot."""

    def __init__(self, snapshot: str = "train"):
        self.snapshot = snapshot
        self.dir = path("features", snapshot)

    # ---- write ----------------------------------------------------------------------------------
    def materialize(self, inter: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame, as_of: pd.Timestamp, cfg: dict) -> dict:
        self.dir.mkdir(parents=True, exist_ok=True)
        item_f = compute_item_features(inter, items, as_of, cfg)
        user_f = compute_user_features(inter, users, items, item_f, as_of, cfg)
        profile = compute_user_genre_profile(inter, items)
        self._check_registry(user_f, item_f)

        user_f.to_parquet(self.dir / "user_features.parquet", index=False)
        item_f.to_parquet(self.dir / "item_features.parquet", index=False)
        profile.reset_index().to_parquet(self.dir / "user_genre_profile.parquet", index=False)
        meta = {
            "snapshot": self.snapshot, "as_of": as_of.isoformat(), "created_at": datetime.now(timezone.utc).isoformat(),
            "source_interactions": len(inter), "user_rows": len(user_f), "item_rows": len(item_f),
            "cold_start_users": int(user_f["is_cold_start"].sum()),
            "n_user_features": len(USER_FEATURES), "n_item_features": len(ITEM_FEATURES), "n_user_item_features": len(USER_ITEM_FEATURES),
            "data_fingerprint": hashlib.md5(pd.util.hash_pandas_object(inter[["user_id", "item_id", "rating", "timestamp"]], index=False).values.tobytes()).hexdigest(),
        }
        write_json(meta, self.dir / "metadata.json")
        return meta

    @staticmethod
    def _check_registry(user_f: pd.DataFrame, item_f: pd.DataFrame) -> None:
        expected_u = {f.name for f in USER_FEATURES} | {"user_id"}
        expected_i = {f.name for f in ITEM_FEATURES} | {"item_id"}
        assert set(user_f.columns) == expected_u, f"unregistered/missing user features: {set(user_f.columns) ^ expected_u}"
        assert set(item_f.columns) == expected_i, f"unregistered/missing item features: {set(item_f.columns) ^ expected_i}"
        warm = user_f[~user_f["is_cold_start"]]
        assert not warm.isna().any().any(), "null features for users with history"
        # Recency is undefined for users without history; everything else must be populated.
        assert not user_f.drop(columns=["user_days_since_last_interaction"]).isna().any().any(), "null user features"

    # ---- read -----------------------------------------------------------------------------------
    @property
    def metadata(self) -> dict:
        return read_json(self.dir / "metadata.json")

    def user_features(self, user_ids=None) -> pd.DataFrame:
        df = pd.read_parquet(self.dir / "user_features.parquet").set_index("user_id")
        return df if user_ids is None else df.reindex(user_ids)

    def item_features(self, item_ids=None) -> pd.DataFrame:
        df = pd.read_parquet(self.dir / "item_features.parquet").set_index("item_id")
        return df if item_ids is None else df.reindex(item_ids)

    def user_genre_profile(self) -> pd.DataFrame:
        return pd.read_parquet(self.dir / "user_genre_profile.parquet").set_index("user_id")

    def genre_matrix(self, item_ids) -> np.ndarray:
        return self.item_features(item_ids)[[f"genre_{g}" for g in GENRES]].to_numpy(dtype=float)

    def user_item_features(self, user_ids, item_ids) -> pd.DataFrame:
        """On-demand pairwise features computed from the stored user and item features."""
        user_ids, item_ids = np.asarray(user_ids), np.asarray(item_ids)
        uf, itf = self.user_features(), self.item_features()
        prof = self.user_genre_profile().reindex(uf.index).fillna(0.0)
        g = itf.loc[item_ids, [f"genre_{x}" for x in GENRES]].to_numpy(dtype=float)
        p = prof.loc[user_ids, GENRES].to_numpy()
        affinity = (g * p).sum(axis=1) / np.maximum(g.sum(axis=1), 1)
        return pd.DataFrame({
            "user_id": user_ids, "item_id": item_ids, "ui_genre_affinity": affinity,
            "ui_popularity_gap": itf.loc[item_ids, "item_log_popularity"].to_numpy() - uf.loc[user_ids, "user_mainstreamness"].to_numpy(),
            "ui_release_year_gap": itf.loc[item_ids, "item_release_year"].astype(float).to_numpy() - uf.loc[user_ids, "user_avg_release_year"].to_numpy(),
        })

    def genre_affinity_matrix(self, user_ids, item_ids) -> np.ndarray:
        """Dense (users x items) ui_genre_affinity - vectorised version used for ranking."""
        prof = self.user_genre_profile().reindex(user_ids).fillna(0.0)[GENRES].to_numpy()
        g = self.genre_matrix(item_ids)
        return prof @ (g / np.maximum(g.sum(axis=1, keepdims=True), 1)).T


def write_catalog() -> None:
    rows = catalog_rows()
    pd.DataFrame(rows).to_csv(path("reports", "feature_catalog.csv"), index=False)
    genre_rows = [r for r in rows if r["name"].startswith("genre_")]
    other = [r for r in rows if not r["name"].startswith("genre_")]
    lines = ["# Feature Catalog", "",
             "All features are generated by `src/features/feature_store.py` from the registry in `src/features/registry.py`.",
             "Snapshots are point-in-time: computed only from interactions at or before the snapshot `as_of` time.", "",
             "| # | Feature | Entity | Type | Description | Logic | Source | Used by |", "|---:|---|---|---|---|---|---|---|"]
    for i, r in enumerate(other, 1):
        lines.append(f"| {i} | `{r['name']}` | {r['entity']} | {r['dtype']} | {r['description']} | `{r['logic']}` | {r['source']} | {r['used_by']} |")
    lines.append(f"| {len(other) + 1} | `genre_<g>` x {len(genre_rows)} | item | int | One-hot genre flags ({', '.join(GENRES)}) | one-hot | items | content model, diversity |")
    lines += ["", f"**Total:** {len(USER_FEATURES)} user, {len(ITEM_FEATURES)} item ({len(genre_rows)} genre flags), {len(USER_ITEM_FEATURES)} user-item features.",
              "", "Storage: `data/features/<snapshot>/{user_features,item_features,user_genre_profile}.parquet` + `metadata.json`."]
    path("reports", "feature_catalog.md").write_text("\n".join(lines))


def run() -> dict:
    cfg = load_config()
    users = pd.read_parquet(path("processed", "users.parquet"))
    items = pd.read_parquet(path("processed", "items.parquet"))
    splits = path("processed", "splits")
    prep = read_json(path("reports", "preparation_report.json"))
    snapshots = {
        "validation": (pd.read_parquet(splits / "train_inner.parquet"), pd.Timestamp(prep["validation_split"]["cutoff_utc"])),
        "train": (pd.read_parquet(splits / "train.parquet"), pd.Timestamp(prep["test_split"]["cutoff_utc"])),
    }
    out = {}
    for name, (inter, as_of) in snapshots.items():
        meta = FeatureStore(name).materialize(inter, users, items, as_of, cfg)
        out[name] = meta
        log.info("Snapshot %-10s as_of=%s users=%d (cold=%d) items=%d fingerprint=%s",
                 name, as_of.date(), meta["user_rows"], meta["cold_start_users"], meta["item_rows"], meta["data_fingerprint"][:10])
    write_catalog()
    fs = FeatureStore("train")
    sample = fs.user_features().query("not is_cold_start").head(3)
    log.info("Sample user features:\n%s", sample[["user_rating_count", "user_avg_rating", "user_preferred_genre", "user_days_since_last_interaction", "user_mainstreamness"]].round(2).to_string())
    log.info("Sample user-item features:\n%s", fs.user_item_features([1, 1, 2], [50, 100, 181]).round(3).to_string(index=False))
    log.info("Feature catalog -> reports/feature_catalog.{md,csv} (%d registered features)", len(catalog_rows()))
    write_json(out, path("reports", "feature_snapshots.json"))
    return out


if __name__ == "__main__":
    run()
