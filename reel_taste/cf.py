"""Collaborative filtering from MovieLens: biases + truncated SVD item factors,
with the user folded in by ridge regression (the user is never in the training matrix)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds

from .config import ARTIFACTS
from . import movielens

K = 96
MIN_ITEM_RATINGS = 20
LAMBDA_I = 25.0
LAMBDA_U = 10.0


class CF:
    def __init__(self, mu, item_ids, item_bias, item_count, item_mean, Q):
        self.mu = float(mu)
        self.item_ids = item_ids
        self.pos = {int(m): i for i, m in enumerate(item_ids)}
        self.item_bias = item_bias
        self.item_count = item_count
        self.item_mean = item_mean
        self.Q = Q
        n = np.linalg.norm(Q, axis=1, keepdims=True)
        self.Qn = Q / np.maximum(n, 1e-9)

    # ---- persistence -------------------------------------------------
    @classmethod
    def load(cls) -> "CF":
        z = np.load(ARTIFACTS / "cf.npz")
        return cls(z["mu"], z["item_ids"], z["item_bias"], z["item_count"], z["item_mean"], z["Q"])

    def save(self):
        np.savez_compressed(ARTIFACTS / "cf.npz", mu=self.mu, item_ids=self.item_ids,
                            item_bias=self.item_bias, item_count=self.item_count,
                            item_mean=self.item_mean, Q=self.Q)

    # ---- training ----------------------------------------------------
    @classmethod
    def train(cls, log=print) -> "CF":
        r = movielens.ratings()
        log(f"ratings: {len(r):,}")
        cnt = r["movieId"].value_counts()
        keep = cnt[cnt >= MIN_ITEM_RATINGS].index.values
        r = r[r["movieId"].isin(keep)]
        item_ids = np.sort(keep).astype(np.int32)
        ii = np.searchsorted(item_ids, r["movieId"].values)
        uu, users = pd.factorize(r["userId"].values)
        y = r["rating"].values.astype(np.float64)
        mu = y.mean()
        n_i = np.bincount(ii, minlength=len(item_ids))
        s_i = np.bincount(ii, weights=y - mu, minlength=len(item_ids))
        b_i = s_i / (n_i + LAMBDA_I)
        res = y - mu - b_i[ii]
        b_u = np.bincount(uu, weights=res) / (np.bincount(uu) + LAMBDA_U)
        res = res - b_u[uu]
        log(f"items kept: {len(item_ids):,}  users: {len(users):,}; running SVD k={K}")
        M = sp.csr_matrix((res.astype(np.float32), (uu, ii)), shape=(len(users), len(item_ids)))
        _, s, vt = svds(M, k=K)
        Q = (vt.T * np.sqrt(s)).astype(np.float32)
        item_mean = (np.bincount(ii, weights=y, minlength=len(item_ids)) / np.maximum(n_i, 1)).astype(np.float32)
        cf = cls(mu, item_ids, b_i.astype(np.float32), n_i.astype(np.int32), item_mean, Q)
        cf.save()
        return cf

    # ---- user fold-in --------------------------------------------------
    def fold_in(self, movie_ids, ratings, weights=None, lam: float = 3.0):
        """Return (b_u, p_u) fitted to the user's ratings on known items."""
        idx = [self.pos.get(int(m)) for m in movie_ids]
        mask = np.array([i is not None for i in idx])
        if mask.sum() < 3:
            return 0.0, np.zeros(self.Q.shape[1], dtype=np.float32)
        ix = np.array([i for i in idx if i is not None])
        y = np.asarray(ratings, dtype=np.float64)[mask]
        w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=np.float64)[mask]
        base = self.mu + self.item_bias[ix]
        b_u = float(np.sum(w * (y - base)) / (w.sum() + LAMBDA_U))
        res = y - base - b_u
        Q = self.Q[ix].astype(np.float64)
        A = (Q * w[:, None]).T @ Q + lam * np.eye(Q.shape[1])
        p = np.linalg.solve(A, (Q * w[:, None]).T @ res)
        return b_u, p.astype(np.float32)

    def predict(self, b_u, p_u, movie_ids) -> np.ndarray:
        out = np.full(len(movie_ids), np.nan)
        for j, m in enumerate(movie_ids):
            i = self.pos.get(int(m)) if m is not None and not pd.isna(m) else None
            if i is not None:
                out[j] = self.mu + self.item_bias[i] + b_u + float(self.Q[i] @ p_u)
        return out

    def vec(self, movie_id):
        if movie_id is None or pd.isna(movie_id):
            return None
        i = self.pos.get(int(movie_id))
        return None if i is None else self.Qn[i]
