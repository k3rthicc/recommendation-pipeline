"""Capture Prefect and MLflow UI screenshots for the final report.

Requires a running Prefect server (port 4200) and MLflow UI (port 5055) plus Playwright
(``pip install playwright``; it drives the locally installed Google Chrome):

    python reports/build/ui_screenshots.py <prefect_flow_run_id>
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "screenshots"
PREFECT, MLFLOW = "http://127.0.0.1:4200", "http://127.0.0.1:5055"


def mlflow_get(endpoint: str) -> dict:
    with urllib.request.urlopen(f"{MLFLOW}/api/2.0/mlflow/{endpoint}") as resp:
        return json.load(resp)


def shot(page, url: str, name: str, wait_text: str | None = None, settle: float = 4.0, click: list[str] | None = None, full: bool = False):
    page.goto(url)
    if wait_text:
        page.get_by_text(wait_text, exact=False).first.wait_for(timeout=30000)
    time.sleep(settle)
    for label in click or []:
        page.get_by_text(label, exact=True).first.click()
        time.sleep(settle)
    page.screenshot(path=str(OUT / name), full_page=full)
    print("saved", name)


def dismiss_prefect_modal(page) -> None:
    skip = page.get_by_role("button", name="Skip")
    if skip.count():
        skip.first.click()
        time.sleep(1)


def main(flow_run_id: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    registry = json.loads((ROOT / "models" / "registry.json").read_text())
    runs = {k: v["run_id"] for k, v in registry["models"].items()}
    exp_id = mlflow_get("experiments/get-by-name?experiment_name=dm4ml-recommendation-pipeline")["experiment"]["experiment_id"]

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 950}, device_scale_factor=1.5)

        # ---------------- Prefect ----------------
        page.goto(f"{PREFECT}/runs/flow-run/{flow_run_id}")
        time.sleep(3)
        dismiss_prefect_modal(page)
        shot(page, f"{PREFECT}/runs/flow-run/{flow_run_id}", "prefect_01_flow_run_graph.png", "Completed", settle=6)
        shot(page, f"{PREFECT}/runs/flow-run/{flow_run_id}", "prefect_02_task_runs.png", "Completed", click=["Task Runs"])
        shot(page, f"{PREFECT}/runs/flow-run/{flow_run_id}", "prefect_03_artifacts.png", "Completed", click=["Artifacts"])
        shot(page, f"{PREFECT}/runs", "prefect_04_runs_list.png", "recommendation-pipeline", settle=5)

        # ---------------- MLflow ----------------
        page.goto(f"{MLFLOW}/#/experiments/{exp_id}/runs")
        time.sleep(5)
        close = page.get_by_role("button", name="Close")
        if close.count():
            close.first.click()
            time.sleep(1)
        final = "&searchFilter=tags.stage%20%3D%20%27final%27"
        shot(page, f"{MLFLOW}/#/experiments/{exp_id}/runs", "mlflow_01_experiment_runs.png", "hybrid", settle=5)
        for name, query in [("mlflow_02_ndcg_charts.png", "ndcg_at_10"), ("mlflow_03_beyond_accuracy_charts.png", "test_"),
                            ("mlflow_04_rmse_charts.png", "rmse")]:
            page.goto(f"{MLFLOW}/#/experiments/{exp_id}/runs?compareRunsMode=CHART{final}")
            time.sleep(5)
            page.get_by_placeholder("Search metric charts").fill(query)
            time.sleep(4)
            page.screenshot(path=str(OUT / name))
            print("saved", name)
        page.goto(f"{MLFLOW}/#/experiments/{exp_id}/runs/{runs['biased_mf']}/model-metrics")
        time.sleep(5)
        page.get_by_placeholder("Search metric charts").fill("train_rmse")
        time.sleep(4)
        page.screenshot(path=str(OUT / "mlflow_05_mf_training_curve.png"))
        shot(page, f"{MLFLOW}/#/experiments/{exp_id}/runs/{runs['hybrid']}", "mlflow_06_hybrid_run_overview.png", "hybrid", settle=5)
        shot(page, f"{MLFLOW}/#/models/movielens-recommender", "mlflow_07_model_registry.png", "movielens-recommender", settle=5)
        browser.close()


if __name__ == "__main__":
    main(sys.argv[1])
