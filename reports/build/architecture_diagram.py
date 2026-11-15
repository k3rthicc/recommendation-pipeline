"""Draws reports/figures/architecture.png (pipeline architecture diagram)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[2] / "reports" / "figures" / "architecture.png"

STAGES = [
    ("1. Ingestion", "MovieLens-100K\n+ 78 injected\nanomalies\nCSV + SQLite", "#4C72B0"),
    ("2. Validation", "33 rules\nerror / warning\nPandera contract\nquality gate", "#C44E52"),
    ("3. Preparation", "dedupe, impute,\nstandardise\ntime-based split\n80/20 (+val)", "#55A868"),
    ("4. Transformation", "user activity\nitem popularity\nuser-item CSR\ngenre / daily", "#8172B2"),
    ("5. Feature store", "49 registered\nfeatures\npoint-in-time\nsnapshots", "#CCB974"),
    ("6. Training", "Popularity, kNN,\nPureSVD, MF,\nHybrid\ntuned on val", "#64B5CD"),
    ("7. Evaluation", "P/R/NDCG@10\nCoverage,\nDiversity\nRMSE / MAE", "#DD8452"),
]
STORES = ["data/raw/", "data/errors/\nreports/validation_*", "data/processed/\n(+ splits/)", "data/processed/\nderived/",
          "data/features/\n<snapshot>/", "models/*.pkl\nregistry.json", "reports/\nevaluation_*"]


def box(ax, x, y, w, h, color, title, body, title_size=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec="none", alpha=0.95))
    ax.text(x + w / 2, y + h - 0.18, title, ha="center", va="top", fontsize=title_size, color="white", weight="bold")
    if body:
        ax.text(x + w / 2, y + h / 2 - 0.15, body, ha="center", va="center", fontsize=8, color="white")


def arrow(ax, x1, y1, x2, y2, color="#333", style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=12, color=color, lw=1.4, linestyle=ls))


def main():
    fig, ax = plt.subplots(figsize=(16, 7.2))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    w, h, gap, x0, y = 1.85, 2.0, 0.3, 0.35, 3.3

    ax.add_patch(FancyBboxPatch((0.15, 6.15), 15.7, 0.8, boxstyle="round,pad=0.02", fc="#1f2937", ec="none"))
    ax.text(8, 6.55, "Orchestration: Prefect flow  dags/recommendation_flow.py  (task dependencies, retries, validation quality gate, weekly cron via --serve)",
            ha="center", va="center", color="white", fontsize=10.5, weight="bold")

    for n, ((title, body, color), store) in enumerate(zip(STAGES, STORES)):
        x = x0 + n * (w + gap)
        box(ax, x, y, w, h, color, title, body)
        arrow(ax, x + w / 2, 6.13, x + w / 2, y + h + 0.03, color="#9ca3af", ls="--")
        if n:
            arrow(ax, x - gap + 0.02, y + h / 2, x - 0.02, y + h / 2)
        ax.add_patch(FancyBboxPatch((x + 0.05, 1.75), w - 0.1, 0.9, boxstyle="round,pad=0.02", fc="#f3f4f6", ec="#9ca3af"))
        ax.text(x + w / 2, 2.2, store, ha="center", va="center", fontsize=7.6, family="monospace")
        arrow(ax, x + w / 2, y - 0.02, x + w / 2, 2.67, color="#6b7280")

    ax.text(0.4, 1.53, "Storage zones (Parquet / CSV / SQLite)", fontsize=8.5, color="#374151", style="italic", va="center")
    box(ax, 0.35, 0.2, 7.3, 1.1, "#0f766e", "Experiment tracking: MLflow (models/mlflow.db)",
        "params, val/test metrics, training curves, lineage tags, artefacts,\nmodel registry: movielens-recommender", title_size=9.5)
    box(ax, 8.0, 0.2, 7.6, 1.1, "#7c3aed", "Serving: src/predict.py + MLflow pyfunc",
        "reads the SAME feature-store snapshot as training (train/serve consistency)\ncold-start users -> demographic segment popularity", title_size=9.5)
    xt, xe = x0 + 5 * (w + gap) + w / 2, x0 + 6 * (w + gap) + w / 2
    arrow(ax, xt, 1.73, 5.5, 1.37, color="#0f766e")
    arrow(ax, xe, 1.73, 6.5, 1.37, color="#0f766e")
    arrow(ax, x0 + 4 * (w + gap) + w / 2, 1.73, 10.5, 1.37, color="#7c3aed")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170)
    print("saved", OUT)


if __name__ == "__main__":
    main()
