"""Recommender models.

All models share one interface (``fit`` / ``recommend`` / ``predict``) so that training,
evaluation and serving treat them identically.

* :class:`PopularityRecommender`   - non-personalised baseline (most-rated items)
* :class:`ItemKNNRecommender`      - item-based collaborative filtering (cosine, top-N neighbours)
* :class:`PureSVDRecommender`      - truncated-SVD matrix factorisation optimised for top-N ranking
* :class:`MatrixFactorization`     - biased MF (Funk-SVD) trained with mini-batch SGD on explicit ratings
* :class:`HybridRecommender`       - weighted blend of PureSVD + item-kNN + biased MF + content (genre affinity
                                     from the feature store) + popularity, with demographic popularity for
                                     cold-start users
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.features.feature_store import FeatureStore


def _zscore_rows(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    return (x - mu) / np.where(sd == 0, 1, sd)


class Recommender:
    name = "base"
    supports_rating_prediction = False

    def fit(self, train: pd.DataFrame, fs: FeatureStore, item_ids: np.ndarray) -> "Recommender":
        self.item_ids = np.asarray(item_ids)
        self.item_index = {int(i): j for j, i in enumerate(self.item_ids)}
        self.user_ids = np.sort(train["user_id"].unique())
        self.user_index = {int(u): j for j, u in enumerate(self.user_ids)}
        self.global_mean = float(train["rating"].mean())
        item_f = fs.item_features(self.item_ids)
        self.popularity = item_f["item_rating_count"].to_numpy(dtype=float)
        self.item_bayes = item_f["item_bayesian_avg"].to_numpy(dtype=float)
        rows = train["user_id"].map(self.user_index).to_numpy()
        cols = train["item_id"].map(self.item_index).to_numpy()
        self.R = sp.csr_matrix((train["rating"].to_numpy(dtype=np.float32), (rows, cols)), shape=(len(self.user_ids), len(self.item_ids)))
        self._fit(train, fs)
        return self

    # ---- hooks for subclasses -------------------------------------------------------------------
    def _fit(self, train: pd.DataFrame, fs: FeatureStore) -> None:
        pass

    def _score_warm(self, uidx: np.ndarray) -> np.ndarray:
        return np.tile(self.popularity, (len(uidx), 1))

    def _score_cold(self, user_ids: np.ndarray) -> np.ndarray:
        """Users without training history fall back to global popularity."""
        return np.tile(self.popularity, (len(user_ids), 1))

    def _predict_warm(self, uidx: np.ndarray, iidx: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # ---- public API -----------------------------------------------------------------------------
    def score(self, user_ids) -> np.ndarray:
        user_ids = np.asarray(user_ids)
        warm = np.array([u in self.user_index for u in user_ids], dtype=bool)
        scores = np.empty((len(user_ids), len(self.item_ids)))
        if warm.any():
            uidx = np.array([self.user_index[u] for u in user_ids[warm]])
            s = self._score_warm(uidx).astype(float)
            seen = self.R[uidx].toarray() > 0
            s[seen] = -np.inf
            scores[warm] = s
        if (~warm).any():
            scores[~warm] = self._score_cold(user_ids[~warm])
        return scores

    def recommend(self, user_ids, k: int = 10) -> dict[int, list[int]]:
        user_ids = np.asarray(user_ids)
        scores = self.score(user_ids)
        top = np.argpartition(-scores, k, axis=1)[:, :k]
        order = np.take_along_axis(scores, top, axis=1).argsort(axis=1)[:, ::-1]
        top = np.take_along_axis(top, order, axis=1)
        return {int(u): self.item_ids[t].tolist() for u, t in zip(user_ids, top)}

    def predict(self, user_ids, item_ids) -> np.ndarray:
        """Predicted explicit rating for (user, item) pairs; unknown users/items use bias fallbacks."""
        user_ids, item_ids = np.asarray(user_ids), np.asarray(item_ids)
        out = np.full(len(user_ids), self.global_mean)
        iidx = np.array([self.item_index.get(int(i), -1) for i in item_ids])
        known_item = iidx >= 0
        out[known_item] = self.item_bayes[iidx[known_item]]
        uidx = np.array([self.user_index.get(int(u), -1) for u in user_ids])
        ok = (uidx >= 0) & known_item
        if ok.any() and type(self)._predict_warm is not Recommender._predict_warm:
            out[ok] = self._predict_warm(uidx[ok], iidx[ok])
        return np.clip(out, 1, 5)

    def params(self) -> dict:
        return {}


class PopularityRecommender(Recommender):
    """Recommends the most-rated unseen items; predicts ratings with the item's Bayesian average."""

    name = "popularity"
    supports_rating_prediction = True

    def __init__(self, min_ratings: int = 10):
        self.min_ratings = min_ratings

    def _fit(self, train, fs):
        self.rank_score = np.where(self.popularity >= self.min_ratings, self.popularity + self.item_bayes / 10, 0.0)

    def _score_warm(self, uidx):
        return np.tile(self.rank_score, (len(uidx), 1))

    def _score_cold(self, user_ids):
        return np.tile(self.rank_score, (len(user_ids), 1))

    def params(self):
        return {"min_ratings": self.min_ratings}


class ItemKNNRecommender(Recommender):
    """Item-based CF: cosine similarity on the binary interaction matrix with shrinkage, top-N neighbours."""

    name = "item_knn"
    supports_rating_prediction = True

    def __init__(self, neighbours: int = 40, shrinkage: float = 10.0):
        self.neighbours, self.shrinkage = neighbours, shrinkage

    def _fit(self, train, fs):
        X = (self.R > 0).astype(np.float32)
        co = (X.T @ X).toarray()
        norms = np.sqrt(np.diag(co))
        sim = co / (np.outer(norms, norms) + self.shrinkage + 1e-9)
        np.fill_diagonal(sim, 0.0)
        k = min(self.neighbours, sim.shape[0] - 1)
        keep = np.argpartition(-sim, k, axis=0)[:k]          # top-k neighbours j of every target item i (column)
        S = np.zeros_like(sim)
        np.put_along_axis(S, keep, np.take_along_axis(sim, keep, axis=0), axis=0)
        self.S = S.astype(np.float32)
        self.X = X
        counts = np.diff(self.R.indptr)
        self.user_mean = np.asarray(self.R.sum(axis=1)).ravel() / np.maximum(counts, 1)

    def _score_warm(self, uidx):
        return np.asarray(self.X[uidx] @ self.S)

    def _predict_warm(self, uidx, iidx):
        preds = np.empty(len(uidx))
        for u in np.unique(uidx):
            m = uidx == u
            row = self.R[u].toarray().ravel()
            rated = row > 0
            centered = np.where(rated, row - self.user_mean[u], 0.0)
            num = centered @ self.S[:, iidx[m]]
            den = rated.astype(float) @ np.abs(self.S[:, iidx[m]])
            preds[m] = self.user_mean[u] + np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        return preds

    def params(self):
        return {"neighbours": self.neighbours, "shrinkage": self.shrinkage}


class PureSVDRecommender(Recommender):
    """Top-N matrix factorisation (PureSVD, Cremonesi et al. 2010).

    Truncated SVD of the user-item rating matrix with unobserved entries as zeros, R ~ U S V^T.
    The score of item i for user u is the reconstruction r_u V V^T. It models *what* a user rates
    rather than the rating value, which is why it ranks better than rating-prediction MF.
    """

    name = "pure_svd"

    def __init__(self, n_factors: int = 10, seed: int = 42):
        self.n_factors, self.seed = n_factors, seed

    def _fit(self, train, fs):
        from sklearn.utils.extmath import randomized_svd

        _, self.singular_values, vt = randomized_svd(self.R.astype(np.float64), self.n_factors, random_state=self.seed)
        self.V = vt.T

    def _score_warm(self, uidx):
        return np.asarray(self.R[uidx] @ self.V) @ self.V.T

    def params(self):
        return {"n_factors": self.n_factors}


class MatrixFactorization(Recommender):
    """Biased matrix factorisation r_ui = mu + b_u + b_i + p_u.q_i trained with mini-batch SGD."""

    name = "biased_mf"
    supports_rating_prediction = True

    def __init__(self, n_factors: int = 50, reg: float = 0.05, lr: float = 0.01, epochs: int = 40, batch_size: int = 128, seed: int = 42):
        self.n_factors, self.reg, self.lr, self.epochs, self.batch_size, self.seed = n_factors, reg, lr, epochs, batch_size, seed

    def _fit(self, train, fs):
        rng = np.random.default_rng(self.seed)
        n_u, n_i = self.R.shape
        self.P = rng.normal(0, 0.1, (n_u, self.n_factors))
        self.Q = rng.normal(0, 0.1, (n_i, self.n_factors))
        self.bu, self.bi = np.zeros(n_u), np.zeros(n_i)
        coo = self.R.tocoo()
        u_all, i_all, r_all = coo.row, coo.col, coo.data.astype(float)
        self.train_rmse_curve = []
        for _ in range(self.epochs):
            perm = rng.permutation(len(r_all))
            for start in range(0, len(perm), self.batch_size):
                b = perm[start:start + self.batch_size]
                u, i, r = u_all[b], i_all[b], r_all[b]
                pu, qi = self.P[u], self.Q[i]
                err = r - (self.global_mean + self.bu[u] + self.bi[i] + (pu * qi).sum(axis=1))
                np.add.at(self.bu, u, self.lr * (err - self.reg * self.bu[u]))
                np.add.at(self.bi, i, self.lr * (err - self.reg * self.bi[i]))
                np.add.at(self.P, u, self.lr * (err[:, None] * qi - self.reg * pu))
                np.add.at(self.Q, i, self.lr * (err[:, None] * pu - self.reg * qi))
            pred = self.global_mean + self.bu[u_all] + self.bi[i_all] + (self.P[u_all] * self.Q[i_all]).sum(axis=1)
            self.train_rmse_curve.append(float(np.sqrt(np.mean((r_all - pred) ** 2))))
        unseen = np.diff(self.R.tocsc().indptr) == 0
        self.Q[unseen] = 0.0                                    # items never rated carry no learned signal

    def _score_warm(self, uidx):
        return self.global_mean + self.bu[uidx, None] + self.bi[None, :] + self.P[uidx] @ self.Q.T

    def predict(self, user_ids, item_ids):
        user_ids, item_ids = np.asarray(user_ids), np.asarray(item_ids)
        uidx = np.array([self.user_index.get(int(u), -1) for u in user_ids])
        iidx = np.array([self.item_index.get(int(i), -1) for i in item_ids])
        out = np.full(len(user_ids), self.global_mean)
        ku, ki = uidx >= 0, iidx >= 0
        out[ki] += self.bi[iidx[ki]]
        out[ku] += self.bu[uidx[ku]]
        both = ku & ki
        out[both] += (self.P[uidx[both]] * self.Q[iidx[both]]).sum(axis=1)
        return np.clip(out, 1, 5)

    def params(self):
        return {"n_factors": self.n_factors, "reg": self.reg, "lr": self.lr, "epochs": self.epochs, "batch_size": self.batch_size}


class HybridRecommender(Recommender):
    """Weighted hybrid of collaborative, content and popularity signals.

    score = z(PureSVD) + knn_weight*z(item-kNN) + mf_weight*z(biased MF)
            + content_weight*z(ui_genre_affinity) + pop_weight*z(log popularity)

    ``z`` standardises each signal per user. Content and popularity signals are read from the feature
    store, so the same features feed training, evaluation and serving. Cold-start users get the
    most-liked items of their demographic segment (age band x gender), backing off to gender and then
    global popularity. Rating prediction is delegated to the biased-MF component.
    """

    name = "hybrid"
    supports_rating_prediction = True

    def __init__(self, svd: PureSVDRecommender, knn: ItemKNNRecommender, mf: MatrixFactorization, knn_weight=0.5,
                 mf_weight=0.25, content_weight=0.25, pop_weight=0.0, min_segment_users=5):
        self.svd, self.knn, self.mf = svd, knn, mf
        self.knn_weight, self.mf_weight, self.content_weight, self.pop_weight = knn_weight, mf_weight, content_weight, pop_weight
        self.min_segment_users = min_segment_users

    def _fit(self, train, fs):
        self.fs_snapshot = fs.snapshot
        self.affinity = fs.genre_affinity_matrix(self.user_ids, self.item_ids)
        self.log_pop = np.log1p(self.popularity)
        users = fs.user_features()[["age_bucket", "gender"]]
        self.user_segments = users
        pos = train[train["rating"] >= 4].join(users, on="user_id")
        self.segment_pop = {}
        for keys in (["age_bucket", "gender"], ["gender"]):
            for seg, grp in pos.groupby(keys):
                if grp["user_id"].nunique() >= self.min_segment_users:
                    counts = grp["item_id"].value_counts()
                    vec = np.zeros(len(self.item_ids))
                    vec[[self.item_index[i] for i in counts.index]] = counts.to_numpy()
                    self.segment_pop[seg if isinstance(seg, tuple) else (seg,)] = vec

    def _score_warm(self, uidx):
        score = _zscore_rows(self.svd._score_warm(uidx))
        for weight, signal in ((self.knn_weight, self.knn._score_warm), (self.mf_weight, self.mf._score_warm),
                               (self.content_weight, lambda u: self.affinity[u]),
                               (self.pop_weight, lambda u: np.tile(self.log_pop, (len(u), 1)))):
            if weight:
                score += weight * _zscore_rows(signal(uidx))
        return score

    def _score_cold(self, user_ids):
        out = np.empty((len(user_ids), len(self.item_ids)))
        for n, u in enumerate(user_ids):
            seg = self.user_segments.loc[u] if u in self.user_segments.index else None
            vec = None
            if seg is not None:
                vec = self.segment_pop.get((seg["age_bucket"], seg["gender"]), self.segment_pop.get((seg["gender"],)))
            out[n] = (vec if vec is not None else self.popularity) + 1e-3 * self.popularity
        return out

    def predict(self, user_ids, item_ids):
        return self.mf.predict(user_ids, item_ids)

    def params(self):
        return {"knn_weight": self.knn_weight, "mf_weight": self.mf_weight, "content_weight": self.content_weight,
                "pop_weight": self.pop_weight, "min_segment_users": self.min_segment_users,
                "cold_start": "demographic segment popularity (age_bucket x gender)", "rating_model": "biased_mf"}
