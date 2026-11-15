# Evidence Index

This file maps every requirement of the assignment to the code that implements it and the artefact or screenshot that proves it ran.
All screenshots were captured from actual pipeline executions on 2026-11-15. Console screenshots are rendered from the captured logs
in [`logs/console/`](logs/console), and UI screenshots come from the local Prefect server and MLflow UI.

## 1. Requirement checklist

| # | Requirement (assignment text) | Implementation | Proof |
|---|---|---|---|
| 1 | Ingest user, item and interaction data from a public dataset | `src/ingestion/ingest.py` | `data/raw/*.csv`, [`ingestion_manifest.json`](ingestion_manifest.json), Fig. 1 |
| 2 | Checks for missing values, duplicates, incorrect types, invalid ratings, invalid timestamps, referential integrity | `src/validation/validate.py` (33 rules + Pandera contracts) | [`validation_report.md`](validation_report.md), Fig. 2 |
| 3 | Invalid records logged / separated into an error dataset | `quarantine()` in `validate.py` | `data/errors/*_errors.csv` (with `error_reasons`), Fig. 3 |
| 4 | Clean data, handle missing values and duplicates, standardise formats | `src/preparation/prepare.py` | [`preparation_report.json`](preparation_report.json), Fig. 4 |
| 5 | Train/test datasets using a **time-based split** | `temporal_split()` + leakage assertion | `data/processed/splits/`, Fig. 4, unit test `test_temporal_split_has_no_leakage` |
| 6 | Derived datasets: user activity summary, item popularity, user-item matrix | `src/transformation/transform.py` | `data/processed/derived/`, [`transformation_summary.json`](transformation_summary.json), Fig. 5 |
| 7 | Create and document user, item and interaction features | `src/features/registry.py`, `feature_store.py` | [`feature_catalog.md`](feature_catalog.md) (49 features), Fig. 6 |
| 8 | Features reusable for training **and** prediction | `FeatureStore` API used by `train.py`, `evaluate.py`, `predict.py` | Fig. 9 (prediction reads the `train` snapshot) |
| 9 | Baseline popularity recommender | `PopularityRecommender` | Fig. 7, Fig. 8 |
| 10 | Personalised recommender (CF / MF / hybrid) | `ItemKNNRecommender`, `PureSVDRecommender`, `MatrixFactorization`, `HybridRecommender` | Fig. 7, Fig. 8 |
| 11 | Track parameters, metrics and artefacts with MLflow | `train.py`, `evaluate.py` | Figs. 14-20 |
| 12 | Precision@K, Recall@K, NDCG@K, Coverage, Diversity (K=10); RMSE and MAE for explicit ratings | `src/evaluation/metrics.py`, `evaluate.py` | [`model_comparison.md`](model_comparison.md), Fig. 8 |
| 13 | Compare models in a results table | `results_table()` | [`evaluation_results.md`](evaluation_results.md), [`model_comparison.md`](model_comparison.md) |
| 14 | Orchestration with Prefect / Airflow, reproducible, correct dependencies | `dags/recommendation_flow.py` | Figs. 10-13 |
| D1 | Complete source code organised in modules | `src/<stage>/` (suggested structure) | [README section 2](../README.md#repository-structure-as-suggested-in-the-assignment) |
| D2 | README with setup and execution instructions | [`README.md`](../README.md) | sections 3-4 |
| D3 | `requirements.txt` | [`requirements.txt`](../requirements.txt) | pinned versions |
| D4 | Architecture diagram | [`figures/architecture.png`](figures/architecture.png) | README section 2 |
| D5 | Data validation report or logs | [`validation_report.md`](validation_report.md), [`logs/pipeline.log`](logs/pipeline.log) | Fig. 2 |
| D6 | Feature catalog in table format | [`feature_catalog.md`](feature_catalog.md), [`feature_catalog.csv`](feature_catalog.csv) | |
| D7 | Model evaluation results comparing the models | [`model_comparison.md`](model_comparison.md) | Fig. 8 |
| D8 | Orchestration workflow file + screenshot of a successful run | `dags/recommendation_flow.py` | Fig. 10 |
| D9 | Short final report (2-3 pages) | [`final_report.pdf`](final_report.pdf) | |

## 2. Data-quality evidence (injected vs detected)

| Table | Injected anomalies | Quarantined (error) | Warnings fixed in preparation | Detected |
|---|---:|---:|---:|---|
| users | 14 | 2 (missing ID, non-integer ID) | 12 (3 dup, 3 missing age, 2 invalid age, 2 invalid gender, 2 bad format) | 14 / 14 |
| items | 9 | 1 (non-integer ID) | 8 (2 dup, 3 missing date, 1 invalid date, 2 bad title format) | 9 / 9 |
| interactions | 55 | 35 (8 invalid rating, 5 non-numeric, 4 missing rating, 6 bad timestamp, 6 unknown user, 6 unknown item) | 20 (15 exact dup, 5 conflicting re-ratings) | 55 / 55 |

The validation report shows a few more warnings than were injected (for example 4 missing release dates and 3 missing IMDb URLs).
These are real gaps in the original MovieLens data (item 267 "unknown") and were also handled.

## 3. Reproducibility evidence

| Check | Result |
|---|---|
| Run 1: stage-by-stage CLI (`reports/build/capture_console.sh`) vs run 2: Prefect flow (`sticky-scorpion`) | max absolute metric difference **0.0** |
| Unit tests | 8 / 8 passed (Fig. 21) |
| Lineage tags on every final MLflow run | `git_commit`, `data_fingerprint`, `feature_snapshot`, `feature_as_of`, `split` (Fig. 19) |

## 4. Screenshots

### Pipeline stages (console)
**Fig. 1 - Ingestion:** source tables read, 78 anomalies injected, raw CSV + SQLite written, manifest with checksums.
![](screenshots/stage_01_ingestion.png)

**Fig. 2 - Validation:** per-rule failure counts by severity, Pandera contracts PASS, quality gate status.
![](screenshots/stage_02_validation.png)

**Fig. 3 - Error dataset:** quarantined interaction records with their `error_reasons`.
![](screenshots/stage_02b_error_dataset.png)

**Fig. 4 - Preparation:** every cleaning action counted, time-based split with leakage check.
![](screenshots/stage_03_preparation.png)

**Fig. 5 - Transformation:** derived datasets (activity summary, popularity, CSR matrix) and the processed data layout.
![](screenshots/stage_04_transformation.png)

**Fig. 6 - Feature store:** two point-in-time snapshots, sample user and user-item features, catalog generated.
![](screenshots/stage_05_features.png)

**Fig. 7 - Training:** validation search for each model family and final refit with MLflow run IDs.
![](screenshots/stage_06_training.png)

**Fig. 8 - Evaluation:** results table for all models, best model registered in MLflow.
![](screenshots/stage_07_evaluation.png)

**Fig. 9 - Prediction / serving:** user profile and top-10 lists served from the `train` feature snapshot (warm user 1, cold-start user 945).
![](screenshots/stage_09_prediction.png)

### Orchestration (Prefect)
**Fig. 10 - Successful flow run graph** (8 tasks, all Completed, dependencies visible).
![](screenshots/prefect_01_flow_run_graph.png)

**Fig. 11 - Task runs list.**
![](screenshots/prefect_02_task_runs.png)

**Fig. 12 - Prefect artifacts** published by the flow (validation summary table, model comparison).
![](screenshots/prefect_03_artifacts.png)

**Fig. 13 - Flow runs history** (two successful runs) and CLI output of the flow.
![](screenshots/prefect_04_runs_list.png)
![](screenshots/stage_08_orchestration_cli.png)

### Experiment tracking (MLflow)
**Fig. 14 - Experiment runs** (final models + nested hyper-parameter search).
![](screenshots/mlflow_01_experiment_runs.png)

**Fig. 15 - NDCG@10 charts** (validation and test, overall / warm / cold).
![](screenshots/mlflow_02_ndcg_charts.png)

**Fig. 16 - Test-metric charts** (coverage, diversity, hit rate, ...).
![](screenshots/mlflow_03_beyond_accuracy_charts.png)

**Fig. 17 - RMSE charts.**
![](screenshots/mlflow_04_rmse_charts.png)

**Fig. 18 - Biased MF training curve** (train RMSE per epoch).
![](screenshots/mlflow_05_mf_training_curve.png)

**Fig. 19 - Hybrid run overview** (parameters, tags, lineage).
![](screenshots/mlflow_06_hybrid_run_overview.png)

**Fig. 20 - Model registry:** `movielens-recommender`.
![](screenshots/mlflow_07_model_registry.png)

### Tests
**Fig. 21 - Unit tests.**
![](screenshots/stage_10_unit_tests.png)
