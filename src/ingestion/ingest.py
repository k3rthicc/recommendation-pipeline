"""Stage 1 - Data ingestion.

Downloads the public MovieLens-100K dataset, converts the three source files
(users, items, interactions) to CSV in the raw zone, adds lineage columns and
loads the same data into a SQLite raw store. A small set of documented,
deliberately corrupted records is appended so that the validation stage can be
demonstrated on realistic data-quality problems.
"""

from __future__ import annotations

import io
import sqlite3
import urllib.request
import uuid
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.utils import file_md5, get_logger, load_config, path, write_json

log = get_logger("ingestion")

GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
    "Romance", "Sci-Fi", "Thriller", "War", "Western",
]
USER_COLS = ["user_id", "age", "gender", "occupation", "zip_code"]
ITEM_COLS = ["item_id", "title", "release_date", "video_release_date", "imdb_url"] + GENRES
INTERACTION_COLS = ["user_id", "item_id", "rating", "timestamp"]


def download_dataset(cfg: dict) -> zipfile.ZipFile:
    archive = path("raw", cfg["ingestion"]["archive_name"])
    if not archive.exists():
        log.info("Downloading %s", cfg["ingestion"]["source_url"])
        urllib.request.urlretrieve(cfg["ingestion"]["source_url"], archive)
    else:
        log.info("Using cached archive %s", archive.name)
    log.info("Archive md5=%s size=%.2f MB", file_md5(archive), archive.stat().st_size / 1e6)
    return zipfile.ZipFile(archive)


def read_source_tables(zf: zipfile.ZipFile) -> dict[str, pd.DataFrame]:
    """Read the source files as strings so the raw zone keeps the data exactly as delivered."""

    def read(member: str, sep: str, cols: list[str]) -> pd.DataFrame:
        raw = zf.read(f"ml-100k/{member}").decode("latin-1")
        return pd.read_csv(io.StringIO(raw), sep=sep, names=cols, dtype=str, keep_default_na=False)

    return {
        "users": read("u.user", "|", USER_COLS),
        "items": read("u.item", "|", ITEM_COLS),
        "interactions": read("u.data", "\t", INTERACTION_COLS),
    }


def inject_anomalies(tables: dict[str, pd.DataFrame], seed: int) -> pd.DataFrame:
    """Append corrupted records to each table and return a log describing every injected issue."""
    rng = np.random.default_rng(seed)
    users, items, inter = tables["users"], tables["items"], tables["interactions"]
    injected: list[dict] = []

    def add(table: str, rows: list[dict], issue: str) -> None:
        tables[table] = pd.concat([tables[table], pd.DataFrame(rows)], ignore_index=True)
        injected.extend({"table": table, "issue": issue, "record": str(r)} for r in rows)

    def sample(df: pd.DataFrame, n: int) -> list[dict]:
        return df.iloc[rng.choice(len(df), n, replace=False)].to_dict("records")

    # ---- users -------------------------------------------------------------------------------
    add("users", sample(users, 3), "exact_duplicate")
    add("users", [{**r, "user_id": str(944 + i), "age": ""} for i, r in enumerate(sample(users, 3))], "missing_age")
    add("users", [{**r, "user_id": str(947 + i), "age": a} for i, (r, a) in enumerate(zip(sample(users, 2), ["150", "-3"]))], "invalid_age")
    add("users", [{**r, "user_id": str(949 + i), "gender": "X"} for i, r in enumerate(sample(users, 2))], "invalid_gender")
    add("users", [{**r, "user_id": str(951 + i), "occupation": f"  {r['occupation'].upper()} "} for i, r in enumerate(sample(users, 2))], "non_standard_format")
    add("users", [{**sample(users, 1)[0], "user_id": "abc"}], "invalid_type")
    add("users", [{**sample(users, 1)[0], "user_id": ""}], "missing_key")

    # ---- items -------------------------------------------------------------------------------
    add("items", sample(items, 2), "exact_duplicate")
    add("items", [{**r, "item_id": str(1683 + i), "release_date": ""} for i, r in enumerate(sample(items, 3))], "missing_release_date")
    add("items", [{**r, "item_id": str(1686 + i), "title": f"   {r['title']}  "} for i, r in enumerate(sample(items, 2))], "non_standard_format")
    add("items", [{**r, "item_id": str(1688 + i), "release_date": "31-Feb-1996"} for i, r in enumerate(sample(items, 1))], "invalid_date")
    add("items", [{**sample(items, 1)[0], "item_id": "X12"}], "invalid_type")

    # ---- interactions ------------------------------------------------------------------------
    add("interactions", sample(inter, 15), "exact_duplicate")
    add("interactions", [{**r, "rating": v} for r, v in zip(sample(inter, 8), ["0", "6", "7", "-1", "10", "0", "6", "9"])], "invalid_rating")
    add("interactions", [{**r, "rating": v} for r, v in zip(sample(inter, 5), ["five", "4.x", "NA?", "abc", "*"])], "invalid_type")
    add("interactions", [{**r, "rating": ""} for r in sample(inter, 4)], "missing_rating")
    add("interactions", [{**r, "timestamp": v} for r, v in zip(sample(inter, 6), ["0", "2051222400", "-86400", "not_a_date", "", "315532800"])], "invalid_timestamp")
    add("interactions", [{**r, "user_id": str(9990 + i)} for i, r in enumerate(sample(inter, 6))], "unknown_user_fk")
    add("interactions", [{**r, "item_id": str(5000 + i)} for i, r in enumerate(sample(inter, 6))], "unknown_item_fk")
    conflicting = []
    for r in sample(inter, 5):
        new_rating = str(int(r["rating"]) % 5 + 1)
        conflicting.append({**r, "rating": new_rating, "timestamp": str(int(r["timestamp"]) + 3600)})
    add("interactions", conflicting, "conflicting_duplicate_pair")

    return pd.DataFrame(injected)


def run() -> dict:
    cfg = load_config()
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    tables = read_source_tables(download_dataset(cfg))
    source_counts = {k: len(v) for k, v in tables.items()}
    for name, df in tables.items():
        log.info("Read source table %-12s rows=%-7d cols=%d", name, len(df), df.shape[1])

    anomalies = pd.DataFrame()
    if cfg["ingestion"]["inject_anomalies"]:
        anomalies = inject_anomalies(tables, cfg["ingestion"]["anomaly_seed"])
        anomalies.to_csv(path("raw", "injected_anomalies_log.csv"), index=False)
        log.info("Injected %d documented anomalous records: %s", len(anomalies), anomalies.groupby("table").size().to_dict())

    manifest = {"batch_id": batch_id, "ingested_at": datetime.now(timezone.utc), "source": cfg["ingestion"]["source_url"], "tables": {}}
    ingested_at = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(path("raw", "raw_store.db")) as conn:
        for name, df in tables.items():
            df = df.copy()
            df["_batch_id"] = batch_id
            df["_ingested_at"] = ingested_at
            csv_path = path("raw", f"{name}.csv")
            df.to_csv(csv_path, index=False)
            df.to_sql(f"raw_{name}", conn, if_exists="replace", index=False)
            manifest["tables"][name] = {
                "file": str(csv_path.relative_to(path("raw").parents[1])),
                "source_rows": source_counts[name],
                "ingested_rows": len(df),
                "columns": len(df.columns),
                "md5": file_md5(csv_path),
            }
            log.info("Wrote raw/%s.csv rows=%d md5=%s and SQLite table raw_%s", name, len(df), manifest["tables"][name]["md5"][:10], name)

    write_json(manifest, path("reports", "ingestion_manifest.json"))
    log.info("Ingestion complete. batch_id=%s manifest=reports/ingestion_manifest.json", batch_id)
    return manifest


if __name__ == "__main__":
    run()
