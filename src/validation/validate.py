"""Stage 2 - Data validation.

Every raw record is checked against a set of rules. Each rule has a severity:

* ``error``   - the record cannot be trusted (bad key, wrong type, impossible rating or timestamp,
                broken referential integrity). It is quarantined in ``data/errors/<table>_errors.csv``
                together with the list of failed rules.
* ``warning`` - the record is usable but needs cleaning (duplicates, missing optional attributes,
                non-standard formatting). It passes through and is fixed in the preparation stage.

Records that pass are type-coerced, checked once more against a Pandera schema contract and written
to ``data/validated``. A JSON + Markdown validation report is produced in ``reports/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd
import pandera.pandas as pa

from src.ingestion.ingest import GENRES
from src.utils import get_logger, load_config, path, write_json

log = get_logger("validation")
LINEAGE_COLS = ["_batch_id", "_ingested_at"]


@dataclass
class Rule:
    name: str
    severity: str
    description: str
    mask: pd.Series


def _is_int(s: pd.Series) -> pd.Series:
    return s.str.fullmatch(r"-?\d+").fillna(False)


def _blank(s: pd.Series) -> pd.Series:
    return s.isna() | (s.astype(str).str.strip() == "")


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _duplicate_rules(df: pd.DataFrame, key: list[str], label: str) -> list[Rule]:
    """Duplicate rules are evaluated on non-quarantined rows only (``df`` is the error-free subset)."""
    data_cols = [c for c in df.columns if c not in LINEAGE_COLS + ["_raw_row"]]
    exact = df.duplicated(subset=data_cols, keep="first")
    key_dup = df.duplicated(subset=key, keep=False) & ~df.duplicated(subset=data_cols, keep=False)
    return [
        Rule("exact_duplicate", "warning", f"Identical {label} record appears more than once", exact),
        Rule("conflicting_duplicate_key", "warning", f"Same {'+'.join(key)} with different attribute values", key_dup),
    ]


def user_rules(df: pd.DataFrame, cfg: dict) -> list[Rule]:
    v = cfg["validation"]
    age = _to_num(df["age"])
    return [
        Rule("missing_user_id", "error", "user_id is null/blank", _blank(df["user_id"])),
        Rule("invalid_type_user_id", "error", "user_id is not a positive integer", ~_blank(df["user_id"]) & ~_is_int(df["user_id"])),
        Rule("missing_age", "warning", "age is null/blank", _blank(df["age"])),
        Rule("invalid_age", "warning", f"age outside [{v['age_min']}, {v['age_max']}] or non-numeric",
             ~_blank(df["age"]) & (age.isna() | (age < v["age_min"]) | (age > v["age_max"]))),
        Rule("invalid_gender", "warning", f"gender not in {v['valid_genders']}", ~df["gender"].isin(v["valid_genders"])),
        Rule("missing_occupation", "warning", "occupation is null/blank", _blank(df["occupation"])),
        Rule("non_standard_occupation", "warning", "occupation has whitespace/upper-case",
             df["occupation"].fillna("") != df["occupation"].fillna("").str.strip().str.lower()),
        Rule("missing_zip_code", "warning", "zip_code is null/blank", _blank(df["zip_code"])),
    ]


def item_rules(df: pd.DataFrame, cfg: dict) -> list[Rule]:
    parsed = pd.to_datetime(df["release_date"], format="%d-%b-%Y", errors="coerce")
    genre_bad = ~df[GENRES].isin(["0", "1"]).all(axis=1)
    return [
        Rule("missing_item_id", "error", "item_id is null/blank", _blank(df["item_id"])),
        Rule("invalid_type_item_id", "error", "item_id is not a positive integer", ~_blank(df["item_id"]) & ~_is_int(df["item_id"])),
        Rule("missing_title", "error", "title is null/blank", _blank(df["title"])),
        Rule("invalid_genre_flags", "error", "genre indicator not in {0,1}", genre_bad),
        Rule("missing_release_date", "warning", "release_date is null/blank", _blank(df["release_date"])),
        Rule("invalid_release_date", "warning", "release_date not a valid DD-Mon-YYYY date", ~_blank(df["release_date"]) & parsed.isna()),
        Rule("non_standard_title", "warning", "title has leading/trailing whitespace", df["title"].fillna("") != df["title"].fillna("").str.strip()),
        Rule("missing_imdb_url", "warning", "imdb_url is null/blank", _blank(df["imdb_url"])),
    ]


def interaction_rules(df: pd.DataFrame, cfg: dict, valid_users: set[int], valid_items: set[int]) -> list[Rule]:
    v = cfg["validation"]
    rating = _to_num(df["rating"])
    ts = _to_num(df["timestamp"])
    ts_min = pd.Timestamp(v["timestamp_min"], tz="UTC").timestamp()
    ts_max = pd.Timestamp(v["timestamp_max"], tz="UTC").timestamp()
    uid, iid = _to_num(df["user_id"]), _to_num(df["item_id"])
    rules = [Rule(f"missing_{c}", "error", f"{c} is null/blank", _blank(df[c])) for c in ["user_id", "item_id", "rating", "timestamp"]]
    rules += [
        Rule("invalid_type_user_id", "error", "user_id is not an integer", ~_blank(df["user_id"]) & ~_is_int(df["user_id"])),
        Rule("invalid_type_item_id", "error", "item_id is not an integer", ~_blank(df["item_id"]) & ~_is_int(df["item_id"])),
        Rule("invalid_type_rating", "error", "rating is not numeric", ~_blank(df["rating"]) & rating.isna()),
        Rule("invalid_rating", "error", f"rating not an integer in [{v['rating_min']}, {v['rating_max']}]",
             rating.notna() & ((rating < v["rating_min"]) | (rating > v["rating_max"]) | (rating % 1 != 0))),
        Rule("invalid_timestamp", "error", f"timestamp non-numeric or outside [{v['timestamp_min']}, {v['timestamp_max']})",
             ~_blank(df["timestamp"]) & (ts.isna() | (ts < ts_min) | (ts >= ts_max))),
        Rule("unknown_user_fk", "error", "user_id not present in validated users (referential integrity)", uid.notna() & ~uid.isin(valid_users)),
        Rule("unknown_item_fk", "error", "item_id not present in validated items (referential integrity)", iid.notna() & ~iid.isin(valid_items)),
    ]
    return rules


USER_SCHEMA = pa.DataFrameSchema(
    {
        "user_id": pa.Column(int, pa.Check.gt(0)),
        "age": pa.Column(float, nullable=True),
        "gender": pa.Column(str, nullable=True),
        "occupation": pa.Column(str, nullable=True),
        "zip_code": pa.Column(str, nullable=True),
    },
    strict=False,
)
ITEM_SCHEMA = pa.DataFrameSchema(
    {
        "item_id": pa.Column(int, pa.Check.gt(0)),
        "title": pa.Column(str, pa.Check.str_length(min_value=1)),
        "release_date": pa.Column("datetime64[ns]", nullable=True),
        **{g: pa.Column(int, pa.Check.isin([0, 1])) for g in GENRES},
    },
    strict=False,
)


def interaction_schema(cfg: dict) -> pa.DataFrameSchema:
    v = cfg["validation"]
    return pa.DataFrameSchema(
        {
            "user_id": pa.Column(int, pa.Check.gt(0)),
            "item_id": pa.Column(int, pa.Check.gt(0)),
            "rating": pa.Column(int, pa.Check.in_range(v["rating_min"], v["rating_max"])),
            "timestamp": pa.Column(int),
            "event_time": pa.Column(pd.DatetimeTZDtype(tz="UTC"), [
                pa.Check.ge(pd.Timestamp(v["timestamp_min"], tz="UTC")),
                pa.Check.lt(pd.Timestamp(v["timestamp_max"], tz="UTC")),
            ]),
        },
        strict=False,
    )


def apply_rules(name: str, df: pd.DataFrame, rules: list[Rule], dup_key: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = df.reset_index(drop=True)
    df.insert(0, "_raw_row", df.index + 2)  # 1-based line number in the raw CSV (header = line 1)
    errors = [r for r in rules if r.severity == "error"]
    error_mask = pd.concat([r.mask.reset_index(drop=True) for r in errors], axis=1).any(axis=1)
    for dup in _duplicate_rules(df[~error_mask], dup_key, name.rstrip("s")):
        dup.mask = dup.mask.reindex(df.index, fill_value=False)
        rules.append(dup)

    def reasons(rule_list: list[Rule]) -> pd.Series:
        parts = pd.DataFrame({r.name: r.mask.reset_index(drop=True).map({True: r.name, False: ""}) for r in rule_list})
        return parts.apply(lambda row: ";".join(x for x in row if x), axis=1)

    invalid = df[error_mask].copy()
    invalid["error_reasons"] = reasons(errors)[error_mask]
    invalid["quarantined_at"] = datetime.now(timezone.utc).isoformat()
    valid = df[~error_mask].copy()

    warnings = [r for r in rules if r.severity == "warning"]
    warn_reasons = reasons(warnings)[~error_mask]
    flagged = valid[warn_reasons != ""].copy()
    flagged["warning_reasons"] = warn_reasons[warn_reasons != ""]

    rule_summary = [
        {"table": name, "rule": r.name, "severity": r.severity, "description": r.description,
         "failed_rows": int(r.mask.sum()), "failed_pct": round(100 * r.mask.mean(), 4)}
        for r in rules
    ]
    stats = {
        "input_rows": len(df), "valid_rows": len(valid), "quarantined_rows": len(invalid),
        "rows_with_warnings": len(flagged), "pass_rate_pct": round(100 * len(valid) / len(df), 3), "rules": rule_summary,
    }
    invalid.to_csv(path("errors", f"{name}_errors.csv"), index=False)
    flagged.to_csv(path("errors", f"{name}_warnings.csv"), index=False)
    log.info("%-12s input=%-7d valid=%-7d quarantined=%-4d warnings=%d",
             name, len(df), len(valid), len(invalid), len(flagged))
    for r in rule_summary:
        if r["failed_rows"]:
            log.info("    [%-7s] %-28s failed=%d", r["severity"].upper(), r["rule"], r["failed_rows"])
    return valid, invalid, stats


def profile(df: pd.DataFrame) -> list[dict]:
    return [
        {"column": c, "dtype": str(df[c].dtype), "nulls": int(df[c].isna().sum()), "unique": int(df[c].nunique(dropna=True))}
        for c in df.columns if not c.startswith("_")
    ]


def coerce_users(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    v = cfg["validation"]
    out = df.copy()
    out["user_id"] = out["user_id"].astype(int)
    out["age"] = _to_num(out["age"]).where(lambda a: a.between(v["age_min"], v["age_max"])).astype(float)
    for c in ["gender", "occupation", "zip_code"]:
        out[c] = out[c].where(~_blank(out[c]), None)
    return out


def coerce_items(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["item_id"] = out["item_id"].astype(int)
    out["release_date"] = pd.to_datetime(out["release_date"], format="%d-%b-%Y", errors="coerce").astype("datetime64[ns]")
    out["video_release_date"] = out["video_release_date"].where(~_blank(out["video_release_date"]), None)
    out["imdb_url"] = out["imdb_url"].where(~_blank(out["imdb_url"]), None)
    out[GENRES] = out[GENRES].astype(int)
    return out


def coerce_interactions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["user_id", "item_id", "rating", "timestamp"]:
        out[c] = out[c].astype(int)
    out["event_time"] = pd.to_datetime(out["timestamp"], unit="s", utc=True).astype("datetime64[ns, UTC]")
    return out


def write_markdown_report(report: dict) -> None:
    lines = [
        "# Data Validation Report", "",
        f"*Generated:* {report['generated_at']}  ",
        f"*Ingestion batch:* `{report['batch_id']}`  ",
        f"*Overall status:* **{report['status']}**", "",
        "## Summary", "",
        "| Table | Input rows | Valid rows | Quarantined | Rows with warnings | Pass rate | Pandera contract |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for t, s in report["tables"].items():
        lines.append(f"| {t} | {s['input_rows']} | {s['valid_rows']} | {s['quarantined_rows']} | {s['rows_with_warnings']} | {s['pass_rate_pct']}% | {s['schema_contract']} |")
    lines += ["", "## Rule results", "", "| Table | Rule | Severity | Failed rows | Description |", "|---|---|---|---:|---|"]
    for t, s in report["tables"].items():
        for r in s["rules"]:
            lines.append(f"| {t} | `{r['rule']}` | {r['severity']} | {r['failed_rows']} | {r['description']} |")
    lines += ["", "Errors are quarantined in `data/errors/<table>_errors.csv` (with an `error_reasons` column).",
              "Warnings are listed in `data/errors/<table>_warnings.csv` and resolved in the preparation stage.", "",
              "## Column profile of validated data", ""]
    for t, s in report["tables"].items():
        lines += [f"### {t}", "", "| Column | dtype | Nulls | Unique |", "|---|---|---:|---:|"]
        lines += [f"| {p['column']} | {p['dtype']} | {p['nulls']} | {p['unique']} |" for p in s["profile"]]
        lines.append("")
    path("reports", "validation_report.md").write_text("\n".join(lines))


def run() -> dict:
    cfg = load_config()
    raw = {t: pd.read_csv(path("raw", f"{t}.csv"), dtype=str, keep_default_na=False) for t in ["users", "items", "interactions"]}
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "batch_id": raw["users"]["_batch_id"].iloc[0], "tables": {}}

    users_ok, _, u_stats = apply_rules("users", raw["users"], user_rules(raw["users"], cfg), ["user_id"])
    items_ok, _, i_stats = apply_rules("items", raw["items"], item_rules(raw["items"], cfg), ["item_id"])
    users_ok, items_ok = coerce_users(users_ok, cfg), coerce_items(items_ok)

    inter = raw["interactions"]
    inter_ok, _, x_stats = apply_rules(
        "interactions", inter, interaction_rules(inter, cfg, set(users_ok["user_id"]), set(items_ok["item_id"])),
        ["user_id", "item_id"],
    )
    inter_ok = coerce_interactions(inter_ok)

    status = "PASS"
    for name, df, schema, stats in [
        ("users", users_ok, USER_SCHEMA, u_stats),
        ("items", items_ok, ITEM_SCHEMA, i_stats),
        ("interactions", inter_ok, interaction_schema(cfg), x_stats),
    ]:
        try:
            schema.validate(df, lazy=True)
            stats["schema_contract"] = "PASS"
        except pa.errors.SchemaErrors as exc:
            stats["schema_contract"] = f"FAIL ({len(exc.failure_cases)} cases)"
            status = "FAIL"
        stats["profile"] = profile(df)
        report["tables"][name] = stats
        df.to_parquet(path("validated", f"{name}.parquet"), index=False)
        log.info("Pandera schema contract for %-12s -> %s", name, stats["schema_contract"])

    report["status"] = status
    write_json(report, path("reports", "validation_report.json"))
    write_markdown_report(report)
    log.info("Validation status=%s; reports/validation_report.{json,md}; quarantined records in data/errors/", status)
    if status != "PASS":
        raise RuntimeError("Validated data violates the schema contract - stopping the pipeline")
    return report


if __name__ == "__main__":
    run()
