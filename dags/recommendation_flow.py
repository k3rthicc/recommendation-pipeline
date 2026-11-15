"""Prefect orchestration of the end-to-end recommendation pipeline.

    python dags/recommendation_flow.py            # one run (shows up in the Prefect UI if a server is running)
    python dags/recommendation_flow.py --serve    # long-running deployment, scheduled weekly (Mon 02:00)

Task graph (each task consumes the artefacts of its upstream task):

    ingest -> validate -> prepare -> transform -> features -> train -> evaluate -> publish_summary

* ``validate`` is a quality gate: if the Pandera contract fails the task raises and nothing downstream runs.
* ``ingest`` retries on network failures.
* Every task is idempotent (rewrites its outputs), so a failed run can be re-executed safely.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prefect import flow, get_run_logger, task  # noqa: E402
from prefect.artifacts import create_markdown_artifact, create_table_artifact  # noqa: E402

from src.evaluation import evaluate  # noqa: E402
from src.features import feature_store  # noqa: E402
from src.ingestion import ingest  # noqa: E402
from src.preparation import prepare  # noqa: E402
from src.training import train  # noqa: E402
from src.transformation import transform  # noqa: E402
from src.utils import path, set_seed  # noqa: E402
from src.validation import validate  # noqa: E402


@task(name="1-ingest", retries=2, retry_delay_seconds=10)
def ingest_task() -> dict:
    manifest = ingest.run()
    get_run_logger().info("Ingested tables: %s", {t: v["ingested_rows"] for t, v in manifest["tables"].items()})
    return manifest


@task(name="2-validate")
def validate_task(manifest: dict) -> dict:
    report = validate.run()
    create_table_artifact(
        key="validation-summary",
        table=[{"table": t, "input": s["input_rows"], "valid": s["valid_rows"], "quarantined": s["quarantined_rows"],
                "warnings": s["rows_with_warnings"], "contract": s["schema_contract"]} for t, s in report["tables"].items()],
        description=f"Validation of batch {manifest['batch_id']}",
    )
    return report


@task(name="3-prepare")
def prepare_task(validation_report: dict) -> dict:
    return prepare.run()


@task(name="4-transform")
def transform_task(preparation_report: dict) -> dict:
    return transform.run()


@task(name="5-features")
def features_task(transformation_summary: dict) -> dict:
    return feature_store.run()


@task(name="6-train")
def train_task(feature_snapshots: dict) -> dict:
    return train.run()


@task(name="7-evaluate")
def evaluate_task(registry: dict) -> dict:
    return evaluate.run()


@task(name="8-publish-summary")
def publish_task(results: dict) -> str:
    md = path("reports", "evaluation_results.md").read_text()
    create_markdown_artifact(key="model-comparison", markdown=md, description="Test-period model comparison")
    best = max(results, key=lambda m: results[m]["ndcg_at_10"])
    get_run_logger().info("Best model: %s (NDCG@10=%.4f)", best, results[best]["ndcg_at_10"])
    return best


@flow(name="recommendation-pipeline", log_prints=True)
def recommendation_pipeline() -> str:
    set_seed()
    manifest = ingest_task()
    report = validate_task(manifest)
    prep = prepare_task(report)
    derived = transform_task(prep)
    feats = features_task(derived)
    registry = train_task(feats)
    results = evaluate_task(registry)
    return publish_task(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", help="serve as a scheduled deployment (weekly)")
    args = parser.parse_args()
    if args.serve:
        recommendation_pipeline.serve(name="weekly-retrain", cron="0 2 * * 1")
    else:
        recommendation_pipeline()
