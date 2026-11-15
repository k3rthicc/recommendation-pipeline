"""Ranking, beyond-accuracy and rating metrics (pure functions, unit-tested in ``tests/``)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


def precision_at_k(recs: Sequence, relevant: set, k: int) -> float:
    return len(set(recs[:k]) & relevant) / k


def recall_at_k(recs: Sequence, relevant: set, k: int) -> float:
    return len(set(recs[:k]) & relevant) / len(relevant) if relevant else 0.0


def ndcg_at_k(recs: Sequence, relevant: set, k: int) -> float:
    """Binary-relevance NDCG: DCG = sum rel_i / log2(i + 1), normalised by the ideal ordering."""
    dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(recs[:k]) if item in relevant)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(recs: Sequence, relevant: set, k: int) -> float:
    return float(bool(set(recs[:k]) & relevant))


def catalog_coverage(all_recs: Iterable[Sequence], n_catalog_items: int) -> float:
    """Share of the catalogue that appears in at least one user's top-K list."""
    distinct = set()
    for recs in all_recs:
        distinct.update(recs)
    return len(distinct) / n_catalog_items


def intra_list_diversity(recs: Sequence, item_vectors: dict) -> float:
    """1 - mean pairwise cosine similarity of the items' genre vectors (higher = more diverse)."""
    vecs = np.array([item_vectors[i] for i in recs], dtype=float)
    if len(vecs) < 2:
        return 0.0
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    unit = vecs / np.where(norms == 0, 1, norms)
    sim = unit @ unit.T
    n = len(vecs)
    return float(1 - (sim.sum() - np.trace(sim)) / (n * (n - 1)))


def novelty(recs: Sequence, item_pop_share: dict) -> float:
    """Mean self-information -log2(p(i)) of recommended items (higher = less obvious items)."""
    return float(np.mean([-np.log2(max(item_pop_share.get(i, 0.0), 1e-6)) for i in recs]))


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))
