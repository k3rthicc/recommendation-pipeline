"""Stage 6 - Model training with MLflow experiment tracking.

1. Hyper-parameter selection on the inner temporal validation split (``train_inner`` -> ``val``)
   using the ``validation`` feature snapshot. Every candidate is a nested MLflow run.
   Ranking models are selected by validation NDCG@K, the rating model (biased MF) by validation RMSE.
2. The selected configurations are refit on the full training period with the ``train`` snapshot.
   Each final model gets its own MLflow run with parameters, validation metrics, the training
   curve, lineage tags (data fingerprint, feature snapshot, git commit) and the pickled model.
"""

from __future__ import annotations

import itertools
import pickle
import subprocess
from collections.abc import Callable

import mlflow
import numpy as np
import pandas as pd

from src.evaluation.evaluate import compute_metrics
from src.features.feature_store import FeatureStore
from src.training.models import (
    HybridRecommender,
    ItemKNNRecommender,
    MatrixFactorization,
    PopularityRecommender,
    PureSVDRecommender,
    Recommender,
)
from src.utils import PROJECT_ROOT, get_logger, load_config, path, set_seed, setup_mlflow, write_json

log = get_logger("training")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "uncommitted"


def _float_metrics(res: dict, prefix: str) -> dict:
    return {f"{prefix}{k}": v for k, v in res.items() if isinstance(v, float) and not np.isnan(v)}


def search(name: str, candidates: list[dict], build: Callable[[dict], Recommender], objective: str, minimise: bool,
           train: pd.DataFrame, val: pd.DataFrame, fs: FeatureStore, item_ids: np.ndarray, k: int, thr: int,
           log_nested: bool = True) -> tuple[Recommender, dict, dict, list[dict]]:
    """Fit every candidate on ``train``, score it on ``val`` and keep the best one."""
    best = None
    rows = []
    for params in candidates:
        model = build(params).fit(train, fs, item_ids)
        res, _ = compute_metrics(model, val, fs, k, thr)
        rows.append({**params, **{m: res[m] for m in (f"ndcg_at_{k}", f"precision_at_{k}", f"coverage_at_{k}", "rmse") if m in res}})
        if log_nested:
            with mlflow.start_run(run_name=f"{name}_" + "_".join(f"{p}{v}" for p, v in params.items()), nested=True):
                mlflow.set_tags({"stage": "tuning", "model_type": name})
                mlflow.log_params({"model": name, **model.params()})
                mlflow.log_metrics(_float_metrics(res, "val_"))
        better = best is None or (res[objective] < best[2][objective] if minimise else res[objective] > best[2][objective])
        if better:
            best = (model, params, res)
    log.info("tune %-10s %2d candidates -> best %s  val %s=%.4f", name, len(candidates), best[1], objective, best[2][objective])
    return best[0], best[1], best[2], rows


def grid(space: dict) -> list[dict]:
    keys = list(space)
    return [dict(zip(keys, values)) for values in itertools.product(*(space[k] for k in keys))]


def tune(cfg: dict, train_inner: pd.DataFrame, val: pd.DataFrame, item_ids: np.ndarray) -> tuple[dict, dict]:
    t, seed = cfg["training"], cfg["project"]["seed"]
    k, thr = t["k"], cfg["preparation"]["relevance_threshold"]
    ndcg = f"ndcg_at_{k}"
    fs = FeatureStore("validation")
    chosen, scores = {}, {}
    common = dict(train=train_inner, val=val, fs=fs, item_ids=item_ids, k=k, thr=thr)

    with mlflow.start_run(run_name="hyperparameter_search"):
        mlflow.set_tags({"stage": "tuning", "feature_snapshot": "validation"})
        _, chosen["popularity"], scores["popularity"], _ = search(
            "popularity", [t["popularity"]], lambda p: PopularityRecommender(**p), ndcg, False, **common)
        knn, chosen["item_knn"], scores["item_knn"], _ = search(
            "item_knn", grid({"neighbours": t["item_knn"]["neighbours_grid"], "shrinkage": [t["item_knn"]["shrinkage"]]}),
            lambda p: ItemKNNRecommender(**p), ndcg, False, **common)
        svd, chosen["pure_svd"], scores["pure_svd"], _ = search(
            "pure_svd", grid({"n_factors": t["pure_svd"]["n_factors_grid"]}),
            lambda p: PureSVDRecommender(**p, seed=seed), ndcg, False, **common)
        mf_cfg = t["biased_mf"]
        mf, chosen["biased_mf"], scores["biased_mf"], _ = search(
            "biased_mf", grid({**mf_cfg["search_space"], "lr": [mf_cfg["lr"]], "epochs": [mf_cfg["epochs"]]}),
            lambda p: MatrixFactorization(**p, seed=seed), "rmse", True, **common)
        _, chosen["hybrid"], scores["hybrid"], rows = search(
            "hybrid", grid(t["hybrid"]), lambda p: HybridRecommender(svd, knn, mf, **p), ndcg, False, log_nested=False, **common)

        hybrid_grid = pd.DataFrame(rows).sort_values(ndcg, ascending=False)
        hybrid_grid.to_csv(path("reports", "hybrid_weight_search.csv"), index=False)
        mlflow.log_artifact(str(path("reports", "hybrid_weight_search.csv")))
        mlflow.log_params({f"chosen.{m}.{p}": v for m, ps in chosen.items() for p, v in ps.items()})
        mlflow.log_metrics({f"val_{m}_{ndcg}": s[ndcg] for m, s in scores.items()})
    write_json({"chosen": chosen, "validation_metrics": scores}, path("reports", "tuning_results.json"))
    return chosen, scores


def run() -> dict:
    cfg = load_config()
    seed = set_seed()
    setup_mlflow()
    splits = path("processed", "splits")
    train, train_inner, val, test = (pd.read_parquet(splits / f"{s}.parquet") for s in ("train", "train_inner", "val", "test"))
    item_ids = np.sort(pd.read_parquet(path("processed", "items.parquet"))["item_id"].to_numpy())

    chosen, val_scores = tune(cfg, train_inner, val, item_ids)

    fs = FeatureStore("train")
    pop = PopularityRecommender(**chosen["popularity"]).fit(train, fs, item_ids)
    knn = ItemKNNRecommender(**chosen["item_knn"]).fit(train, fs, item_ids)
    svd = PureSVDRecommender(**chosen["pure_svd"], seed=seed).fit(train, fs, item_ids)
    mf = MatrixFactorization(**chosen["biased_mf"], seed=seed).fit(train, fs, item_ids)
    hybrid = HybridRecommender(svd, knn, mf, **chosen["hybrid"]).fit(train, fs, item_ids)
    final = {"popularity": pop, "item_knn": knn, "pure_svd": svd, "biased_mf": mf, "hybrid": hybrid}

    tags = {"stage": "final", "feature_snapshot": "train", "feature_as_of": fs.metadata["as_of"],
            "data_fingerprint": fs.metadata["data_fingerprint"], "git_commit": git_commit(), "split": "global temporal 80/20"}
    registry = {"models": {}, "feature_snapshot": "train", "chosen_hyperparameters": chosen}
    for name, model in final.items():
        model_path = path("models", f"{name}.pkl")
        with open(model_path, "wb") as fh:
            pickle.dump(model, fh)
        with mlflow.start_run(run_name=name) as run_:
            mlflow.set_tags({**tags, "model_type": name})
            mlflow.log_params({"model": name, "k": cfg["training"]["k"], "seed": seed, "train_rows": len(train), **model.params()})
            mlflow.log_metrics(_float_metrics(val_scores[name], "val_"))
            for step, v in enumerate(getattr(model, "train_rmse_curve", [])):
                mlflow.log_metric("train_rmse", v, step=step)
            mlflow.log_artifact(str(model_path), artifact_path="model_pickle")
            mlflow.log_artifact(str(PROJECT_ROOT / "configs" / "config.yaml"), artifact_path="config")
            mlflow.log_artifact(str(path("features", "train", "metadata.json")), artifact_path="feature_snapshot")
            registry["models"][name] = {"run_id": run_.info.run_id, "path": str(model_path.relative_to(PROJECT_ROOT)), "params": model.params()}
        log.info("trained %-11s params=%s mlflow_run=%s", name, model.params(), run_.info.run_id[:8])

    rel = test[test["rating"] >= cfg["preparation"]["relevance_threshold"]]["user_id"].unique()
    warm = [int(u) for u in sorted(rel) if u in knn.user_index][:2]
    cold = [int(u) for u in sorted(rel) if u not in knn.user_index][:1]
    registry["sample_users"] = warm + cold
    write_json(registry, path("models", "registry.json"))
    log.info("Model registry -> models/registry.json; MLflow experiment '%s' (sqlite:///models/mlflow.db)", cfg["mlflow"]["experiment"])
    return registry


if __name__ == "__main__":
    run()
