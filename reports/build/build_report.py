"""Builds reports/final_report.pdf: a 3-page report followed by the execution-evidence appendix.

    python reports/build/build_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "reports"
SHOTS = REP / "screenshots"
REPO = "https://github.com/k3rthicc/recommendation-pipeline"
ROLL = "2025AG05355"

PAGE_W, PAGE_H = A4
MARGIN = 1.7 * cm
CONTENT_W = PAGE_W - 2 * MARGIN

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=13.5, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#1f3a5f"))
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=11.5, spaceBefore=6, spaceAfter=3, textColor=colors.HexColor("#1f3a5f"))
BODY = ParagraphStyle("Body", parent=ss["BodyText"], fontSize=9.4, leading=12.6, spaceAfter=4)
SMALL = ParagraphStyle("Small", parent=BODY, fontSize=8.2, leading=10.4)
CELL = ParagraphStyle("Cell", parent=BODY, fontSize=7.9, leading=9.8, spaceAfter=0)
CELL_B = ParagraphStyle("CellB", parent=CELL, fontName="Helvetica-Bold")
CAPTION = ParagraphStyle("Caption", parent=SMALL, alignment=TA_CENTER, textColor=colors.HexColor("#444444"), spaceBefore=2, spaceAfter=8)
TITLE = ParagraphStyle("Title", parent=ss["Title"], fontSize=17, leading=21, spaceAfter=4, textColor=colors.HexColor("#1f3a5f"))
SUB = ParagraphStyle("Sub", parent=BODY, alignment=TA_CENTER, fontSize=10, leading=13)


def P(text, style=BODY):
    return Paragraph(text, style)


def bullets(items, style=BODY):
    return ListFlowable([ListItem(P(i, style), leftIndent=10, value="•") for i in items], bulletType="bullet", start="•", leftIndent=12)


def table(rows, widths, header=True, font=7.9, zebra=True, bold_rows=()):
    data = [[c if not isinstance(c, str) else P(c, CELL_B if (header and r == 0) or r in bold_rows else CELL) for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8c2cc")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe8f3")))
    if zebra:
        for r in range(1 if header else 0, len(rows)):
            if r % 2 == 0:
                style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f6f8fa")))
    for r in bold_rows:
        style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#e7f5e9")))
    t.setStyle(TableStyle(style))
    return t


def figure(name, caption, max_h=None, width=CONTENT_W):
    img_path = SHOTS / name if (SHOTS / name).exists() else REP / "figures" / name
    iw, ih = ImageReader(str(img_path)).getSize()
    w = width
    h = w * ih / iw
    max_h = max_h or (PAGE_H - 2 * MARGIN - 3.2 * cm)
    if h > max_h:
        h, w = max_h, max_h * iw / ih
    img = Image(str(img_path), width=w, height=h)
    img.hAlign = "CENTER"
    return KeepTogether([img, P(caption, CAPTION)])


def stage(title, text, shots, lead=()):
    """Evidence block: heading + short description + one or more screenshots (heading kept with the first one)."""
    figs = [figure(name, caption, max_h=max_h) for name, caption, max_h in shots]
    return [KeepTogether([*lead, P(title, H2), P(text), figs[0]]), *figs[1:]]


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(MARGIN, 1.0 * cm, f"DM4ML Assignment 2 | Roll No. {ROLL} | {REPO}")
    canvas.drawRightString(PAGE_W - MARGIN, 1.0 * cm, f"Page {doc.page}")
    canvas.restoreState()


def results_rows():
    df = pd.read_csv(REP / "evaluation_results.csv", index_col=0)
    ev = json.loads((REP / "evaluation_results.json").read_text())["results"]
    label = {"popularity": "Popularity (baseline)", "item_knn": "Item-kNN CF", "pure_svd": "PureSVD (MF)", "biased_mf": "Biased MF", "hybrid": "Hybrid"}
    rows = [["Model", "P@10", "R@10", "NDCG @10", "Hit @10", "Cov. @10", "Div. @10", "RMSE", "MAE", "Warm NDCG", "Cold NDCG"]]
    best = df["ndcg_at_10"].idxmax()
    for m, r in df.iterrows():
        fmt = lambda v: "n/a" if pd.isna(v) else f"{v:.4f}"
        rows.append([label[m], fmt(r.precision_at_10), fmt(r.recall_at_10), fmt(r.ndcg_at_10), fmt(r.hit_rate_at_10),
                     f"{r.coverage_at_10:.3f}", f"{r.diversity_at_10:.3f}", fmt(r.rmse), fmt(r.mae),
                     fmt(ev[m]["warm_ndcg_at_10"]), fmt(ev[m]["cold_ndcg_at_10"])])
    return rows, list(df.index).index(best) + 1, df, ev


def build():
    prep = json.loads((REP / "preparation_report.json").read_text())
    val = json.loads((REP / "validation_report.json").read_text())
    feats = json.loads((REP / "feature_snapshots.json").read_text())
    rows, best_row, df, ev = results_rows()
    t = prep["test_split"]
    q = {k: v["quarantined_rows"] for k, v in val["tables"].items()}
    w = {k: v["rows_with_warnings"] for k, v in val["tables"].items()}
    gain = 100 * (df.loc["hybrid", "ndcg_at_10"] / df.loc["popularity", "ndcg_at_10"] - 1)
    warm_gain = 100 * (ev["hybrid"]["warm_ndcg_at_10"] / ev["popularity"]["warm_ndcg_at_10"] - 1)

    s = []
    # ------------------------------------------------------------------ title
    s += [P("End-to-End Data Management Pipeline for a Recommendation System", TITLE),
          P(f"Data Management for Machine Learning (DM4ML) - Assignment 2 &nbsp;|&nbsp; Roll No. <b>{ROLL}</b>", SUB), Spacer(1, 4)]
    link = Table([[P(f'<b>GitHub repository (public):</b> <link href="{REPO}" color="blue"><u>{REPO}</u></link>', ParagraphStyle("L", parent=BODY, alignment=TA_CENTER, fontSize=10.5))]],
                 colWidths=[CONTENT_W])
    link.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#1f3a5f")), ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eef4fb")),
                              ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    s += [link, Spacer(1, 4),
          P("Key documents in the repository: <b>README.md</b> (process document, setup and run instructions), <b>reports/evidence.md</b> "
            "(requirement-to-proof index), <b>reports/feature_catalog.md</b>, <b>reports/model_comparison.md</b>, "
            "<b>reports/validation_report.md</b>, <b>dags/recommendation_flow.py</b> (Prefect flow).", SMALL)]

    # ------------------------------------------------------------------ 1
    s += [P("1. Objective and data", H1),
          P("The goal is a clean, modular and reproducible pipeline that recommends movies to users of an online movie platform. "
            "It covers ingestion, validation, preparation, transformation, feature management, training, evaluation and orchestration. "
            "Data source: the public <b>MovieLens-100K</b> dataset (943 users with demographics, 1,682 movies with 19 genre flags, "
            "100,000 timestamped 1-5 ratings, Sep 1997 - Apr 1998). Because MovieLens is already clean, ingestion appends <b>78 documented "
            "corrupted records</b> (missing values, duplicates, wrong types, ratings such as 0/6/10, impossible timestamps, unknown "
            "foreign keys, bad formatting). Every one is listed in <i>data/raw/injected_anomalies_log.csv</i>, so the quality stages can be "
            "verified against a known ground truth.")]

    # ------------------------------------------------------------------ 2
    s += [P("2. Architecture", H1), figure("architecture.png", "Figure 1 - Pipeline architecture: stages, storage zones, MLflow tracking, Prefect orchestration and serving.", max_h=7.6 * cm)]

    # ------------------------------------------------------------------ 3
    s += [P("3. Design decisions per stage", H1)]
    dd = [
        ["Stage", "What was built", "Why (design decision)"],
        ["Ingestion", "Download + MD5 checksum; read source files <b>as strings</b>; lineage columns (<i>_batch_id</i>, <i>_ingested_at</i>); raw CSV + SQLite raw store; manifest.",
         "The raw zone keeps data exactly as delivered, so type errors are caught by validation rather than silently coerced by the reader. Checksums and batch IDs give lineage."],
        ["Validation", f"33 rules with two severities. <b>Errors</b> are quarantined to <i>data/errors/&lt;table&gt;_errors.csv</i> with <i>error_reasons</i> "
         f"({q['interactions']} interactions, {q['users']} users, {q['items']} item). <b>Warnings</b> ({sum(w.values())} rows) pass through to be cleaned. A Pandera schema contract is enforced on the output.",
         "Separating 'untrustworthy' from 'fixable' avoids throwing away usable data. Referential integrity is checked against the <i>validated</i> users/items. A contract failure stops the flow (quality gate)."],
        ["Preparation", "De-duplication; conflicting re-ratings -> keep latest; age imputed by occupation median; release date from title year; drop 100%-null column. "
         f"<b>Global time-based split</b> at {t['cutoff_utc'][:10]} (train {t['train_rows']:,} / test {t['test_rows']:,}) plus an inner validation split.",
         "A global cut-off is the strictest leakage-free protocol (future behaviour of other users is never seen). It is asserted in code and unit-tested. The inner split keeps the test set untouched during tuning."],
        ["Transformation", "From the <b>train period only</b>: user activity summary, item popularity (Bayesian average), genre long table, daily activity, sparse user-item CSR matrix (95.0% sparse).",
         "Derived datasets computed only from history cannot leak test information. Parquet is columnar, typed and compact."],
        ["Feature store", f"Registry of <b>49 features</b> (user / item / user-item) with description, logic, source and consumer. <b>Point-in-time snapshots</b> "
         f"(<i>validation</i> as of {feats['validation']['as_of'][:10]}, <i>train</i> as of {feats['train']['as_of'][:10]}) with metadata and data fingerprint.",
         "One API (<i>FeatureStore</i>) is used by training, evaluation and <i>predict.py</i>, which gives train/serve consistency. A registry check blocks undocumented features, and the catalog is generated from code."],
        ["Training", "Popularity baseline; item-kNN CF; PureSVD (MF for ranking); biased MF with SGD (MF for ratings); weighted hybrid with demographic cold-start. "
         "Grid search on validation (54 hybrid weight combinations).",
         "Two MF variants because rating-prediction MF minimises RMSE but ranks poorly. 66% of test users are cold-start under a time split, so the hybrid uses demographic segments from the feature store."],
        ["Evaluation", "P/R/NDCG/Hit@10, catalogue coverage, intra-list genre diversity, novelty, RMSE/MAE. Reported overall and for warm vs cold users. Best model registered in MLflow.",
         "Beyond-accuracy metrics expose popularity bias. The warm/cold split shows <i>where</i> each model helps."],
        ["Orchestration", "Prefect flow, 8 dependent tasks, retries on ingestion, validation quality gate, Prefect artifacts, weekly cron via <i>--serve</i>; plain-Python runner as an alternative.",
         "Prefect runs locally without containers and shows task-level state and logs in its UI. Every task is idempotent, so a failed run can simply be re-executed."],
    ]
    s.append(table(dd, [2.3 * cm, 7.6 * cm, CONTENT_W - 9.9 * cm]))

    # ------------------------------------------------------------------ 4
    s += [P("4. Tools", H1),
          table([["Concern", "Tool"],
                 ["Processing / storage", "Python 3.12, Pandas, NumPy, SciPy (sparse), Parquet (PyArrow), CSV, SQLite"],
                 ["Data validation", "Custom rule engine + <b>Pandera</b> schema contracts"],
                 ["Feature management", "Custom offline feature store (registry, point-in-time snapshots, metadata)"],
                 ["Modelling", "NumPy / SciPy / scikit-learn (randomized SVD) implementations of all five recommenders"],
                 ["Experiment tracking", "<b>MLflow</b> (SQLite backend): params, metrics, curves, artefacts, lineage tags, pyfunc model registry"],
                 ["Orchestration", "<b>Prefect 3</b> (local server UI), Makefile, plain-Python runner"],
                 ["Quality", "pytest (8 unit tests), pinned requirements, fixed seed"]],
                [4 * cm, CONTENT_W - 4 * cm])]

    # ------------------------------------------------------------------ 5
    s += [P("5. Results (held-out future period, K = 10)", H1),
          P(f"Test users with at least one relevant item (rating >= 4): {ev['hybrid']['eval_users']} ({ev['hybrid']['warm_users']} warm, "
            f"{ev['hybrid']['cold_users']} cold-start). All hyper-parameters were chosen on the validation split."),
          table(rows, [3.1 * cm] + [(CONTENT_W - 3.1 * cm) / 10] * 10, bold_rows=(best_row,)),
          Spacer(1, 4),
          bullets([
              f"<b>The hybrid is the best model</b>: NDCG@10 {df.loc['hybrid', 'ndcg_at_10']:.4f} vs {df.loc['popularity', 'ndcg_at_10']:.4f} for the popularity baseline "
              f"(<b>+{gain:.1f}%</b>), with the best precision and recall as well. On warm users it improves NDCG@10 by {warm_gain:.1f}%. On cold-start users, "
              "demographic segment popularity beats global popularity. It is registered as <i>movielens-recommender</i> in MLflow.",
              "<b>Rating prediction and ranking pull in different directions.</b> Biased MF has the lowest RMSE (0.97 on warm users vs 1.04 for the baseline) but the weakest ranking, "
              "because its lists favour rarely rated, highly rated films. PureSVD ranks better, which is why the hybrid blends both.",
              "<b>Accuracy vs diversity:</b> personalised models cover 3.2-3.4x more of the catalogue than the baseline. The hybrid's lower diversity comes "
              "from genre-homogeneous segment lists for cold-start users.",
              "<b>Reproducibility:</b> a stage-by-stage CLI run and a Prefect run produced identical metrics (max difference 0.0)."]),
          ]

    # ------------------------------------------------------------------ 6
    s += [P("6. Limitations and future work", H1),
          bullets([
              "MovieLens-100K is small and dense compared with production data, and the anomalies are synthetic (documented) rather than organic.",
              "Static batch evaluation: models are not refreshed during the 6-week test window, which penalises every model for cold-start users. Incremental retraining (the weekly Prefect schedule) would reduce this.",
              "The genre-affinity content feature is coarse (19 genres) and received zero weight in tuning. Text or embedding features could add signal.",
              "Offline metrics only, and the logged data is popularity-biased. An online A/B test and an online feature store (e.g. Feast + Redis) are the natural next steps.",
          ])]

    # ------------------------------------------------------------------ 7
    s += [P("7. Data-quality and reproducibility evidence", H1),
          P("Every injected anomaly was detected (ground truth: <i>data/raw/injected_anomalies_log.csv</i>). Hard errors were quarantined and "
            "soft issues were repaired in preparation:"),
          table([["Table", "Injected", "Quarantined (error dataset)", "Repaired in preparation (warning)", "Detected"],
                 ["users", "14", "2: missing ID, non-integer ID", "12: 3 duplicates, 5 missing/invalid ages (imputed), 2 invalid genders, 2 badly formatted occupations", "14 / 14"],
                 ["items", "9", "1: non-integer ID", "8: 2 duplicates, 4 missing/invalid release dates (imputed from title), 2 untrimmed titles", "9 / 9"],
                 ["interactions", "55", "35: 8 out-of-range, 5 non-numeric, 4 missing ratings; 6 bad timestamps; 12 unknown user/item IDs",
                  "20: 15 exact duplicates removed, 5 conflicting re-ratings (latest kept)", "55 / 55"]],
                [2.1 * cm, 1.5 * cm, 5.2 * cm, CONTENT_W - 10.6 * cm, 1.8 * cm]),
          Spacer(1, 4),
          bullets([
              "<b>Configuration as code:</b> a single <i>configs/config.yaml</i> holds paths, thresholds, split fractions, search grids and the seed (42). <i>requirements.txt</i> is pinned.",
              "<b>Lineage:</b> ingestion manifest with MD5 checksums, batch IDs on raw rows, a data fingerprint per feature snapshot, and git commit + snapshot tags on every MLflow run.",
              "<b>Determinism:</b> two complete runs (stage-by-stage CLI and Prefect flow) produced identical metrics. All stages are idempotent.",
              "<b>Tests:</b> 8 pytest unit tests (metric formulas, validation rules, no temporal leakage, keep-latest de-duplication, point-in-time features) all pass.",
              "<b>One-command execution:</b> <i>make pipeline</i> or <i>python -m src.pipeline</i>; the Prefect flow and the MLflow UI are described in the README.",
          ])]

    # ------------------------------------------------------------------ Appendix A
    appendix_intro = [P("Appendix A - Execution evidence for every pipeline stage", H1),
                      P("Console screenshots are rendered from the captured output of <i>python -m src.pipeline --stage &lt;name&gt;</i> "
                        "(script: <i>reports/build/capture_console.sh</i>; raw logs in <i>reports/logs/console/</i>). UI screenshots were taken from the "
                        "local Prefect server and MLflow UI after the successful runs.", SMALL)]
    s += [PageBreak()]
    s += stage("A.1 Data ingestion",
               "The three source tables are read, 78 documented anomalies are appended, and raw CSVs plus SQLite tables are written with lineage columns. "
               "The ingestion manifest records row counts and MD5 checksums.",
               [("stage_01_ingestion.png", "Figure A1 - Ingestion run and manifest.", 12 * cm)], lead=appendix_intro)
    s += stage("A.2 Data validation",
               "Rule-level results by severity: errors are quarantined and warnings are passed on for cleaning. All three Pandera contracts pass, so the quality gate opens. "
               "The quarantined records and their reasons are shown in the error dataset below.",
               [("stage_02_validation.png", "Figure A2 - Validation rules, failure counts and Pandera contract status.", 12 * cm),
                ("stage_02b_error_dataset.png", "Figure A3 - Error datasets in data/errors/ with error_reasons per quarantined record.", 9.5 * cm)])
    s += stage("A.3 Data preparation",
               "Every cleaning action is counted. The global time-based split, the inner validation split and the no-leakage assertion are logged.",
               [("stage_03_preparation.png", "Figure A4 - Cleaning actions and time-based split.", 9 * cm)])
    s += stage("A.4 Data transformation",
               "Derived datasets built from the training period: user activity summary, item popularity, genre table, daily activity and the sparse user-item matrix.",
               [("stage_04_transformation.png", "Figure A5 - Derived datasets and the processed/feature data layout.", 13 * cm)])
    s += stage("A.5 Feature management",
               "Two point-in-time feature snapshots are materialised with data fingerprints. Sample user and user-item features are shown, and the catalog is generated from the registry.",
               [("stage_05_features.png", "Figure A6 - Feature-store materialisation.", 10 * cm)])
    s += stage("A.6 Model training",
               "Validation search per model family (nested MLflow runs), followed by a final refit on the full training period. Each final model is logged as its own MLflow run.",
               [("stage_06_training.png", "Figure A7 - Hyper-parameter selection and final training.", 9 * cm)])
    s += stage("A.7 Model evaluation",
               "All models are evaluated on the future test period. The comparison table is printed and the best model (hybrid) is registered in the MLflow model registry.",
               [("stage_07_evaluation.png", "Figure A8 - Evaluation metrics and model comparison.", 10 * cm),
                ("model_comparison.png", "Figure A9 - Model comparison chart (reports/figures/model_comparison.png).", 6.2 * cm)])
    s += stage("A.8 Prediction / serving and unit tests",
               "<i>src/predict.py</i> serves recommendations using the same <i>train</i> feature snapshot as training: a warm user (1) and a cold-start user (945, demographic fallback). "
               "The unit tests cover metrics, validation rules, temporal leakage, de-duplication and point-in-time features.",
               [("stage_09_prediction.png", "Figure A10 - Recommendations with features from the feature store.", 11 * cm),
                ("stage_10_unit_tests.png", "Figure A11 - pytest: 8/8 tests passed.", 6 * cm)])

    # ------------------------------------------------------------------ Appendix B
    s += [PageBreak(), P("Appendix B - Orchestration (Prefect UI)", H1),
          P("The flow <i>recommendation-pipeline</i> (dags/recommendation_flow.py) ran successfully twice. Task dependencies are encoded by passing each task's "
            "output to its downstream task, which the run graph shows as a strict chain.", SMALL)]
    s += [figure("prefect_01_flow_run_graph.png", "Figure B1 - Successful flow run: 8 tasks Completed in 25 s, dependency graph and artifacts.", 11.5 * cm),
          figure("prefect_02_task_runs.png", "Figure B2 - Task runs of the flow (all Completed).", 11 * cm),
          figure("prefect_03_artifacts.png", "Figure B3 - Prefect artifacts published by the flow: validation summary table and model comparison.", 11 * cm),
          figure("prefect_04_runs_list.png", "Figure B4 - Flow-run history: two successful end-to-end runs.", 11 * cm),
          figure("stage_08_orchestration_cli.png", "Figure B5 - Prefect flow run from the command line (task states).", 9 * cm)]

    # ------------------------------------------------------------------ Appendix C
    s += [PageBreak(), P("Appendix C - Experiment tracking (MLflow UI)", H1),
          P("Experiment <i>dm4ml-recommendation-pipeline</i>: nested tuning runs, five final model runs per execution (params, validation and test metrics, "
            "training curves, artefacts, lineage tags) and the registered model.", SMALL)]
    s += [figure("mlflow_01_experiment_runs.png", "Figure C1 - Experiment runs: final models and nested hyper-parameter search.", 11 * cm),
          figure("mlflow_02_ndcg_charts.png", "Figure C2 - NDCG@10 per model (test and validation; overall, warm and cold users).", 11 * cm),
          figure("mlflow_03_beyond_accuracy_charts.png", "Figure C3 - Test metrics: coverage, diversity, hit rate, cold-start metrics.", 11 * cm),
          figure("mlflow_04_rmse_charts.png", "Figure C4 - RMSE metrics per model.", 11 * cm),
          figure("mlflow_05_mf_training_curve.png", "Figure C5 - Biased-MF training curve (train RMSE per epoch).", 11 * cm),
          figure("mlflow_06_hybrid_run_overview.png", "Figure C6 - Hybrid run: metrics, lineage tags (git commit, data fingerprint, feature snapshot) and registered model.", 11 * cm),
          figure("mlflow_07_model_registry.png", "Figure C7 - Model registry: movielens-recommender (one version per pipeline run).", 11 * cm)]

    # ------------------------------------------------------------------ Appendix D
    s += [PageBreak(), P("Appendix D - Deliverables cross-check", H1)]
    checklist = [
        ["Requirement / deliverable", "Location in repository", "Status"],
        ["Ingest user, item, interaction data", "src/ingestion/ingest.py, data/raw/, reports/ingestion_manifest.json", "Done"],
        ["Missing / duplicate / type / rating / timestamp / referential-integrity checks", "src/validation/validate.py, reports/validation_report.md", "Done"],
        ["Invalid records separated into an error dataset", "data/errors/*_errors.csv (error_reasons column)", "Done"],
        ["Cleaning, missing values, duplicates, standard formats", "src/preparation/prepare.py, reports/preparation_report.json", "Done"],
        ["Time-based train/test split", "data/processed/splits/, leakage assertion + unit test", "Done"],
        ["User activity summary, item popularity, user-item matrix", "src/transformation/transform.py, data/processed/derived/", "Done"],
        ["Documented, reusable features (train + predict)", "src/features/, data/features/, reports/feature_catalog.md, src/predict.py", "Done"],
        ["Popularity baseline + personalised models", "src/training/models.py (5 models)", "Done"],
        ["MLflow tracking of params, metrics, artefacts", "src/training/train.py, src/evaluation/evaluate.py, Appendix C", "Done"],
        ["Precision/Recall/NDCG@10, Coverage, Diversity, RMSE, MAE + comparison table", "reports/model_comparison.md, reports/evaluation_results.md", "Done"],
        ["Orchestration with correct dependencies, reproducible", "dags/recommendation_flow.py, Appendix B", "Done"],
        ["Modular source code in the suggested structure", "data/{raw,processed,features,errors}, src/&lt;stage&gt;/, dags/, configs/, models/, reports/", "Done"],
        ["README.md with setup and execution instructions (process document)", "README.md", "Done"],
        ["requirements.txt", "requirements.txt (pinned)", "Done"],
        ["Architecture diagram", "reports/figures/architecture.png, README (Mermaid)", "Done"],
        ["Data validation report / logs", "reports/validation_report.md, reports/logs/", "Done"],
        ["Feature catalog in table format", "reports/feature_catalog.md / .csv", "Done"],
        ["Model evaluation results comparing models", "reports/model_comparison.md", "Done"],
        ["Orchestration workflow file + successful-run screenshot", "dags/recommendation_flow.py, Figure B1", "Done"],
        ["Final report (PDF)", "reports/final_report.pdf (this document)", "Done"],
        ["Evidence index", "reports/evidence.md", "Done"],
    ]
    s.append(table(checklist, [7.2 * cm, CONTENT_W - 8.7 * cm, 1.5 * cm]))
    s += [Spacer(1, 8), figure("github_repo.png", f"Figure D1 - Public GitHub repository: {REPO}", 12.5 * cm)]

    out = REP / "final_report.pdf"
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=1.4 * cm, bottomMargin=1.6 * cm,
                            title=f"{ROLL} - DM4ML Assignment 2 - Recommendation Data Pipeline", author=f"Roll No {ROLL}",
                            subject="DM4ML Assignment 2")
    doc.build(s, onFirstPage=on_page, onLaterPages=on_page)
    print("saved", out)


if __name__ == "__main__":
    build()
