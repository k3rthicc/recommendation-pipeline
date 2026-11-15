import numpy as np
import pytest

from src.evaluation import metrics as M


def test_precision_recall_hit():
    recs, rel = [1, 2, 3, 4, 5], {2, 5, 9}
    assert M.precision_at_k(recs, rel, 5) == pytest.approx(2 / 5)
    assert M.recall_at_k(recs, rel, 5) == pytest.approx(2 / 3)
    assert M.hit_rate_at_k(recs, rel, 5) == 1.0
    assert M.hit_rate_at_k(recs, {42}, 5) == 0.0


def test_ndcg_perfect_and_partial():
    assert M.ndcg_at_k([1, 2, 3], {1, 2, 3}, 3) == pytest.approx(1.0)
    dcg = 1 / np.log2(3)                      # relevant item at rank 2
    idcg = 1.0                                # one relevant item -> ideal at rank 1
    assert M.ndcg_at_k([7, 1, 8], {1}, 3) == pytest.approx(dcg / idcg)
    assert M.ndcg_at_k([7, 8, 9], {1}, 3) == 0.0


def test_coverage_and_diversity():
    assert M.catalog_coverage([[1, 2], [2, 3]], 10) == pytest.approx(0.3)
    vecs = {1: [1, 0], 2: [1, 0], 3: [0, 1]}
    assert M.intra_list_diversity([1, 2], vecs) == pytest.approx(0.0)
    assert M.intra_list_diversity([1, 3], vecs) == pytest.approx(1.0)


def test_rating_errors():
    assert M.rmse([1, 3], [2, 5]) == pytest.approx(np.sqrt((1 + 4) / 2))
    assert M.mae([1, 3], [2, 5]) == pytest.approx(1.5)
