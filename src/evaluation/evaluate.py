"""Stage 7 - Model evaluation.

Evaluates every trained model on the held-out *future* (test) period:

* ranking:          Precision@K, Recall@K, NDCG@K, HitRate@K   (relevant = test rating >= 4)
* beyond-accuracy:  catalogue Coverage@K, intra-list Diversity@K (genre vectors), Novelty@K
* rating accuracy:  RMSE, MAE on every test rating

Ranking metrics are reported for all test users and separately for warm (seen in training) and
cold-start users. Results are logged to the model's MLflow run, and the best model (by NDCG@K)
is registered in the MLflow model registry.
"""

from __future__ import annotations

import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from src.evaluation import metrics as M
from src.features.feature_store import FeatureStore
from src.training.models import Recommender
from src.utils import PROJECT_ROOT, get_logger, load_config, path, read_json, setup_mlflow, write_json

log = get_logger("evaluation")
RANKING = ["precision", "recall", "ndcg", "hit_rate"]


def compute_metrics(model: Recommender, test: pd.DataFrame, fs: FeatureStore, k: int, threshold: int) -> tuple[dict, dict]:
    relevant = test[test["rating"] >= threshold].groupby("user_id")["item_id"].apply(set)
    users = np.array(sorted(relevant.index))
    recs = model.recommend(users, k)
    warm_users = set(model.user_index)

    item_f = fs.item_features(model.item_ids)
    vectors = dict(zip(model.item_ids, fs.genre_matrix(model.item_ids)))
    pop_share = dict(zip(model.item_ids, item_f["item_rating_count"].to_numpy() / item_f["item_rating_count"].sum()))

    per_user = pd.DataFrame({
        "user_id": users,
        "segment": ["warm" if u in warm_users else "cold" for u in users],
        "precision": [M.precision_at_k(recs[u], relevant[u], k) for u in users],
        "recall": [M.recall_at_k(recs[u], relevant[u], k) for u in users],
        "ndcg": [M.ndcg_at_k(recs[u], relevant[u], k) for u in users],
        "hit_rate": [M.hit_rate_at_k(recs[u], relevant[u], k) for u in users],
        "diversity": [M.intra_list_diversity(recs[u], vectors) for u in users],
        "novelty": [M.novelty(recs[u], pop_share) for u in users],
    })
    out = {f"{m}_at_{k}": float(per_user[m].mean()) for m in RANKING}
    out[f"coverage_at_{k}"] = M.catalog_coverage(recs.values(), len(model.item_ids))
    out[f"diversity_at_{k}"] = float(per_user["diversity"].mean())
    out[f"novelty_at_{k}"] = float(per_user["novelty"].mean())
    for seg in ("warm", "cold"):
        sub = per_user[per_user["segment"] == seg]
        out[f"{seg}_users"] = len(sub)
        for m in ("precision", "recall", "ndcg"):
            out[f"{seg}_{m}_at_{k}"] = float(sub[m].mean()) if len(sub) else float("nan")
    out["eval_users"] = len(users)
    if model.supports_rating_prediction:
        pred = model.predict(test["user_id"].to_numpy(), test["item_id"].to_numpy())
        out["rmse"], out["mae"] = M.rmse(test["rating"], pred), M.mae(test["rating"], pred)
        warm_mask = test["user_id"].isin(warm_users).to_numpy()
        out["rmse_warm"] = M.rmse(test["rating"][warm_mask], pred[warm_mask])
    return out, recs


def results_table(results: dict, k: int) -> pd.DataFrame:
    cols = [f"precision_at_{k}", f"recall_at_{k}", f"ndcg_at_{k}", f"hit_rate_at_{k}", f"coverage_at_{k}",
            f"diversity_at_{k}", f"novelty_at_{k}", "rmse", "mae", f"warm_ndcg_at_{k}", f"cold_ndcg_at_{k}"]
    return pd.DataFrame(results).T[cols].astype(float)


def plot_results(table: pd.DataFrame, k: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    table[[f"precision_at_{k}", f"recall_at_{k}", f"ndcg_at_{k}"]].plot.bar(ax=axes[0], rot=15, title=f"Accuracy @ {k}")
    table[[f"coverage_at_{k}", f"diversity_at_{k}"]].plot.bar(ax=axes[1], rot=15, title=f"Beyond accuracy @ {k}")
    table[["rmse", "mae"]].plot.bar(ax=axes[2], rot=15, title="Rating error (lower is better)")
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path("reports", "figures", "model_comparison.png"), dpi=150)
    plt.close(fig)


def register_best(best: str, registry: dict) -> str | None:
    """Log the best model as an MLflow pyfunc and register it in the model registry."""
    from src.training.pyfunc import RecommenderPyfunc

    run_id = registry["models"][best]["run_id"]
    with mlflow.start_run(run_id=run_id):
        info = mlflow.pyfunc.log_model(
            name="recommender", python_model=RecommenderPyfunc(),
            artifacts={"model": str(path("models", f"{best}.pkl"))}, code_paths=[str(PROJECT_ROOT / "src")],
            input_example=pd.DataFrame({"user_id": [1, 2]}),
            pip_requirements=str(PROJECT_ROOT / "requirements.txt"),
            registered_model_name="movielens-recommender",
        )
    return getattr(info, "registered_model_version", None)


def run() -> dict:
    cfg = load_config()
    k, thr = cfg["training"]["k"], cfg["preparation"]["relevance_threshold"]
    setup_mlflow()
    registry = read_json(path("models", "registry.json"))
    test = pd.read_parquet(path("processed", "splits", "test.parquet"))
    fs = FeatureStore("train")
    items = pd.read_parquet(path("processed", "items.parquet")).set_index("item_id")

    results, samples = {}, []
    for name, entry in registry["models"].items():
        with open(path("models", f"{name}.pkl"), "rb") as fh:
            model = pickle.load(fh)
        res, recs = compute_metrics(model, test, fs, k, thr)
        results[name] = res
        for u in registry["sample_users"]:
            if u in recs:
                samples.append({"model": name, "user_id": u, "rank_titles": " | ".join(items.loc[recs[u][:5], "title"])})
        with mlflow.start_run(run_id=entry["run_id"]):
            mlflow.log_metrics({f"test_{m}": v for m, v in res.items() if isinstance(v, float) and not np.isnan(v)})
        log.info("%-22s P@%d=%.4f R@%d=%.4f NDCG@%d=%.4f Cov=%.3f Div=%.3f%s", name, k, res[f"precision_at_{k}"], k,
                 res[f"recall_at_{k}"], k, res[f"ndcg_at_{k}"], res[f"coverage_at_{k}"], res[f"diversity_at_{k}"],
                 f" RMSE={res['rmse']:.4f} MAE={res['mae']:.4f}" if "rmse" in res else "")

    table = results_table(results, k)
    table.round(4).to_csv(path("reports", "evaluation_results.csv"))
    plot_results(table, k)
    best = table[f"ndcg_at_{k}"].idxmax()
    version = register_best(best, registry)
    pd.DataFrame(samples).to_csv(path("reports", "sample_recommendations.csv"), index=False)

    pretty = table.rename(columns=lambda c: c.replace(f"_at_{k}", f"@{k}"))
    md = ["# Model Evaluation Results", "",
          f"Held-out test period: interactions after the temporal cut-off ({len(test)} ratings). K = {k}; relevant = rating >= {thr}.",
          f"Evaluated users: {results[best]['eval_users']} ({results[best]['warm_users']} warm, {results[best]['cold_users']} cold-start).", "",
          pretty.round(4).to_markdown(), "",
          f"**Best model by NDCG@{k}: `{best}`** (registered in MLflow as `movielens-recommender` v{version}).", "",
          "Personalised models (item-kNN, MF) use global popularity for cold-start users; the hybrid uses demographic segment popularity.",
          "Coverage = share of the catalogue recommended to at least one user; Diversity = 1 - mean pairwise genre cosine similarity within a list.",
          "", "![comparison](figures/model_comparison.png)"]
    path("reports", "evaluation_results.md").write_text("\n".join(md))
    write_json({"k": k, "best_model": best, "registered_version": version, "results": results}, path("reports", "evaluation_results.json"))
    log.info("Results table:\n%s", pretty.round(4).to_string())
    log.info("Best model by NDCG@%d: %s -> registered as movielens-recommender v%s", k, best, version)
    return results


if __name__ == "__main__":
    run()
