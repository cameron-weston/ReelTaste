"""The hybrid ranker: crowd taste (MovieLens CF), content (TMDB), TMDB links and priors,
blended by a model trained only on the user's own ratings."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Chosen on a time-based validation slice inside the training years (see README):
MODE = "centered"   # "residual" won on random folds but lost on the time-based slice
BLEND = "best"      # "best" = the CV winner of ridge/gbm; "avg" = average both
CF_FEATS = ["cf_pred", "cf_knn", "cf_max", "cf_n", "ml_mean", "ml_logn"]
CT_FEATS = ["crowd", "ct_knn", "ct_max", "ct_n", "ct_ridge", "link", "link_n",
            "tmdb_avg", "tmdb_logn", "tmdb_pop", "year", "runtime", "doc", "english", "animation"]
ALL_FEATS = CF_FEATS + CT_FEATS


class Context:
    """Shared, user-independent pieces: CF model, content space, metadata by key."""

    def __init__(self, cf=None, cs=None, meta: dict | None = None):
        self.cf, self.cs, self.meta = cf, cs, meta or {}


# Pseudo-votes pulling small-sample TMDB averages toward the catalog mean. For films the time-based
# validation called 0 vs 300 a tie (0.621 vs 0.620), so films keep the simpler unshrunk average.
# Shows have no offline test and suffer from fan-inflated low-vote averages, so they are shrunk.
TMDB_SHRINK = {"movie": 0, "tv": 300}


def _catalog_means(ctx: Context) -> dict:
    if getattr(ctx, "_means", None) is None:
        means = {}
        for kind in ("movie", "tv"):
            v = [(m.get("vote_average"), m.get("vote_count")) for k, m in ctx.meta.items()
                 if k.startswith(kind + ":") and (m.get("vote_count") or 0) >= 200]
            means[kind] = float(np.average([a for a, _ in v], weights=[np.log1p(n) for _, n in v])) if v else 6.5
        ctx._means = means
    return ctx._means


def crowd_score(ctx: Context, key: str) -> float:
    """TMDB average, shrunk toward the catalog mean by vote count; shows shifted onto the film scale."""
    m = ctx.meta.get(key) or {}
    n = m.get("vote_count") or 0
    if n < 10 or m.get("vote_average") is None:
        return np.nan
    kind = key.split(":")[0]
    means = _catalog_means(ctx)
    mu = means.get(kind, 6.5)
    c = TMDB_SHRINK[kind]
    a = (m["vote_average"] * n + mu * c) / (n + c)
    if kind == "tv":
        a -= means["tv"] - means["movie"]
    return a


def _knn(S: np.ndarray, r_c: np.ndarray, w: np.ndarray, k: int = 30, shrink: float = 1.0):
    """Similarity-weighted average of centred ratings over the top-k positive neighbours."""
    if S.shape[1] == 0:
        n = S.shape[0]
        return np.zeros(n), np.zeros(n), np.zeros(n)
    k = min(k, S.shape[1])
    idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
    s = np.take_along_axis(S, idx, axis=1)
    s = np.clip(s, 0, None) ** 2  # sharpen: near neighbours matter far more
    rr = r_c[idx]
    ww = w[idx]
    num = (s * ww * rr).sum(1)
    den = (s * ww).sum(1)
    return num / (den + shrink * 0.05), S.max(1), (S > 0.35).sum(1)


def features(train: pd.DataFrame, targets: pd.DataFrame, ctx: Context) -> pd.DataFrame:
    """train: key, ml_id, rating, w.  targets: key, ml_id.  Returns one feature row per target."""
    F = pd.DataFrame(index=targets.index, columns=ALL_FEATS, dtype=float)
    r = train["rating"].to_numpy(float)
    w = train["w"].to_numpy(float)
    ubar = float(np.average(r, weights=w))
    r_c = r - ubar

    # Crowd prior: TMDB average calibrated onto this user's scale (fitted on the train rows only).
    def tmdb_avg(keys):
        return np.array([crowd_score(ctx, k) for k in keys], float)

    a_tr, a_tg = tmdb_avg(train["key"]), tmdb_avg(targets["key"])
    ok = ~np.isnan(a_tr)
    if ok.sum() >= 10:
        coef = np.polyfit(a_tr[ok], r[ok], 1, w=np.sqrt(w[ok]))
        crowd_tr = np.where(ok, np.polyval(coef, np.nan_to_num(a_tr)), ubar)
        F["crowd"] = np.where(np.isnan(a_tg), np.nan, np.polyval(coef, np.nan_to_num(a_tg)))
    else:
        crowd_tr = np.full(len(r), ubar)
    if MODE == "residual":
        r_c = r - crowd_tr  # taste signals learn how far you depart from the crowd

    cf = ctx.cf
    if cf is not None:
        b_u, p_u = cf.fold_in(train["ml_id"].fillna(-1).to_numpy(), r, w)
        F["cf_pred"] = cf.predict(b_u, p_u, targets["ml_id"].tolist())
        tv = [cf.vec(m) for m in train["ml_id"]]
        tm = np.array([v is not None for v in tv])
        if tm.any():
            Vt = np.stack([v for v in tv if v is not None])
            gv = [cf.vec(m) for m in targets["ml_id"]]
            gm = np.array([v is not None for v in gv])
            if gm.any():
                S = np.stack([v for v in gv if v is not None]) @ Vt.T
                kn, mx, n = _knn(S, r_c[tm], w[tm])
                F.loc[gm, "cf_knn"] = kn
                F.loc[gm, "cf_max"] = mx
                F.loc[gm, "cf_n"] = n
        pos = [cf.pos.get(int(m)) if pd.notna(m) else None for m in targets["ml_id"]]
        cnt = np.array([cf.item_count[p] if p is not None else np.nan for p in pos], float)
        mean = np.array([cf.item_mean[p] if p is not None else np.nan for p in pos], float)
        F["ml_mean"] = (mean * cnt + cf.mu * 50) / (cnt + 50)
        F["ml_logn"] = np.log1p(cnt)

    cs = ctx.cs
    if cs is not None:
        tr = np.array([cs.index.get(k, -1) for k in train["key"]])
        gr = np.array([cs.index.get(k, -1) for k in targets["key"]])
        tm, gm = tr >= 0, gr >= 0
        if tm.any() and gm.any():
            S = cs.Z[gr[gm]] @ cs.Z[tr[tm]].T
            kn, mx, n = _knn(S, r_c[tm], w[tm])
            F.loc[gm, "ct_knn"] = kn
            F.loc[gm, "ct_max"] = mx
            F.loc[gm, "ct_n"] = n
            rid = Ridge(alpha=2.0, fit_intercept=False)
            rid.fit(cs.X[tr[tm]], r_c[tm], sample_weight=w[tm])
            F.loc[gm, "ct_ridge"] = rid.predict(cs.X[gr[gm]])

    # TMDB recommendation/similar links, in both directions.
    meta = ctx.meta
    if meta:
        tr_ids = {}
        for k, rc in zip(train["key"], r_c):
            tr_ids[k] = rc
        inbound: dict[str, list[float]] = {}
        for k, rc in tr_ids.items():
            m = meta.get(k)
            if m is None:
                continue
            kind = k.split(":")[0]
            for t in list(m.get("recommendations", [])) + list(m.get("similar", []))[:10]:
                inbound.setdefault(f"{kind}:{t}", []).append(rc)
        link, link_n = [], []
        for k in targets["key"]:
            vals = list(inbound.get(k, []))
            m = meta.get(k)
            if m is not None:
                kind = k.split(":")[0]
                for t in list(m.get("recommendations", [])) + list(m.get("similar", []))[:10]:
                    v = tr_ids.get(f"{kind}:{t}")
                    if v is not None:
                        vals.append(v)
            link.append(sum(vals) / (len(vals) + 2) if vals else 0.0)
            link_n.append(len(vals))
        F["link"], F["link_n"] = link, link_n
        rows = [meta.get(k, {}) for k in targets["key"]]
        F["tmdb_avg"] = a_tg
        F["tmdb_raw"] = [m.get("vote_average") if (m.get("vote_count") or 0) >= 10 else np.nan for m in rows]
        F["tmdb_logn"] = [np.log1p(m.get("vote_count") or 0) for m in rows]
        F["tmdb_pop"] = [np.log1p(m.get("popularity") or 0) for m in rows]
        F["year"] = [m.get("year") if m.get("year") is not None and not pd.isna(m.get("year")) else np.nan for m in rows]
        F["runtime"] = [m.get("runtime") or np.nan for m in rows]
        F["doc"] = [float("Documentary" in (m.get("genres") or [])) for m in rows]
        F["animation"] = [float("Animation" in (m.get("genres") or [])) for m in rows]
        F["english"] = [float(m.get("language") == "en") for m in rows]
    F["ubar"] = ubar
    return F.astype(float)


# Features that are themselves rating predictions: a higher value can only mean a higher rating.
SIGNED = ("cf_pred", "cf_knn", "ml_mean", "crowd", "ct_knn", "ct_ridge", "link", "tmdb_avg")


class SignedRidge(BaseEstimator, RegressorMixin):
    """Weighted ridge (unpenalised intercept) whose coefficients on the SIGNED columns can't go
    negative. With ~800 ratings and correlated signals, a plain ridge gives some of them negative
    weights that only fit noise; the constraint beat it on both validation folds."""

    def __init__(self, alpha: float = 100.0, signed=SIGNED):
        self.alpha, self.signed = alpha, signed

    def fit(self, X, y, sample_weight=None):
        names = list(X.columns) if hasattr(X, "columns") else []
        X, y = np.asarray(X, float), np.asarray(y, float)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        xm, ym = np.average(X, axis=0, weights=w), np.average(y, weights=w)
        sw = np.sqrt(w)[:, None]
        p = X.shape[1]
        A = np.vstack([(X - xm) * sw, np.sqrt(self.alpha) * np.eye(p)])
        b = np.concatenate([(y - ym) * sw[:, 0], np.zeros(p)])
        lb = np.array([0.0 if n in self.signed else -np.inf for n in names] or [-np.inf] * p)
        self.coef_ = lsq_linear(A, b, bounds=(lb, np.inf)).x
        self.intercept_ = ym - xm @ self.coef_
        return self

    def predict(self, X):
        return np.asarray(X, float) @ self.coef_ + self.intercept_


def _learners():
    # Weights are rescaled to mean 1 before fitting, so alpha means the same thing on every fit.
    # alpha=100 was best on the validation folds (30 and 300 both lost).
    return {
        "ridge": lambda: Pipeline([("scale", StandardScaler()),
                                   ("ridge", SignedRidge(alpha=100.0))]).set_output(transform="pandas"),
        "gbm": lambda: HistGradientBoostingRegressor(max_depth=3, learning_rate=0.04, max_iter=250,
                                                     min_samples_leaf=25, l2_regularization=1.0,
                                                     random_state=0),
    }


COUNT_FEATS = ("cf_n", "ct_n", "link_n")


def _prep(F: pd.DataFrame, cols: list[str], kind: str, med: pd.Series) -> np.ndarray:
    """med: training-set column medians, so targets never fill gaps from their own statistics."""
    X = F[cols].copy()
    if kind == "ridge":
        # Missing → neutral 0 for centred signals, training median otherwise, plus missing flags.
        flags = X.isna().astype(float).add_suffix("_na")
        X = X.fillna({c: 0.0 for c in cols if c in ("cf_knn", "ct_knn", "ct_ridge", "link")})
        X = X.fillna(med.reindex(cols)).fillna(0.0)
        X = pd.concat([X, flags.loc[:, flags.any()]], axis=1)
        return X
    return X.to_numpy()


class Hybrid:
    """Two routed blends: 'full' for films with crowd data, 'content' for films without."""

    def __init__(self, ctx: Context, folds: int = 5, seed: int = 0):
        self.ctx, self.folds, self.seed = ctx, folds, seed
        self.models: dict[str, tuple] = {}
        self.cv: dict[str, dict] = {}

    def oof_features(self, train: pd.DataFrame) -> pd.DataFrame:
        F = []
        kf = KFold(self.folds, shuffle=True, random_state=self.seed)
        for a, b in kf.split(train):
            F.append(features(train.iloc[a], train.iloc[b], self.ctx))
        return pd.concat(F).loc[train.index]

    def fit(self, train: pd.DataFrame):
        # Mean-1 weights: regularisation strength no longer depends on how many or how old ratings are.
        train = train.assign(w=train["w"] / train["w"].mean())
        self.train = train.copy()
        F = self.oof_features(train)
        y = train["rating"].to_numpy(float) - F["ubar"].to_numpy()
        w = train["w"].to_numpy(float)
        has_content = self.ctx.cs is not None
        routes = {"full": ALL_FEATS if has_content else CF_FEATS, "content": CT_FEATS}
        for route, cols in routes.items():
            cols = [c for c in cols if F[c].notna().any()]
            if not cols:
                continue
            best = None
            for name, make in _learners().items():
                err, kf = [], KFold(5, shuffle=True, random_state=1)
                for a, b in kf.split(F):
                    med = F.iloc[a][cols].median()
                    Xa = _prep(F.iloc[a], cols, name, med)
                    Xb = _prep(F.iloc[b], cols, name, med)
                    if name == "ridge":
                        Xb = Xb.reindex(columns=Xa.columns, fill_value=0.0)
                    m = make()
                    m.fit(Xa, y[a], **({"ridge__sample_weight": w[a]} if name == "ridge" else {"sample_weight": w[a]}))
                    err.append(np.sqrt(np.average((m.predict(Xb) - y[b]) ** 2, weights=w[b])))
                score = float(np.mean(err))
                self.cv.setdefault(route, {})[name] = score
                if best is None or score < best[0]:
                    best = (score, name)
            fitted = []
            for name in ([best[1]] if BLEND == "best" else list(_learners())):
                med = F[cols].median()
                X = _prep(F, cols, name, med)
                m = _learners()[name]()
                m.fit(X, y, **({"ridge__sample_weight": w} if name == "ridge" else {"sample_weight": w}))
                fitted.append((name, cols, m, list(X.columns) if name == "ridge" else None, med))
            self.models[route] = fitted
        return self

    def describe(self) -> dict:
        return {k: "+".join(f[0] for f in v) for k, v in self.models.items()}

    def predict(self, targets: pd.DataFrame, F: pd.DataFrame | None = None) -> pd.DataFrame:
        if F is None:
            F = features(self.train, targets, self.ctx)
        has_cf = F["cf_pred"].notna().to_numpy()
        # Training features came from folds holding (k-1)/k of the ratings, so neighbour/link counts
        # were that much smaller; scale serving counts to match.
        Fm = F.copy()
        for c in COUNT_FEATS:
            if c in Fm:
                Fm[c] = Fm[c] * (self.folds - 1) / self.folds
        out = np.full(len(F), np.nan)
        for route, mask in (("full", has_cf), ("content", ~has_cf)):
            if route not in self.models or not mask.any():
                continue
            preds = []
            for name, cols, m, xcols, med in self.models[route]:
                X = _prep(Fm[mask], cols, name, med)
                if name == "ridge":
                    X = X.reindex(columns=xcols, fill_value=0.0)
                preds.append(m.predict(X))
            out[mask] = np.mean(preds, axis=0) + F.loc[mask, "ubar"].to_numpy()
        # Anything unroutable falls back to the user's average.
        out = np.where(np.isnan(out), F["ubar"].to_numpy(), out)
        F = F.copy()
        F["pred"] = np.clip(out, 0.5, 5.0)
        F["score"] = out  # uncapped, for ranking: several titles can predict above 5 and still differ
        F["route"] = np.where(has_cf, "full", "content")
        support = (has_cf * 0.4 + (F["cf_max"].fillna(0) > 0.35) * 0.25 +
                   (F["ct_max"].fillna(0) > 0.45) * 0.25 + (F["link_n"].fillna(0) > 0) * 0.2)
        conf = np.select([support >= 0.75, support >= 0.45], ["high", "medium"], "low")
        # Without crowd data, few TMDB votes caps confidence.
        votes = np.expm1(F["tmdb_logn"].fillna(0).to_numpy()) if "tmdb_logn" in F else np.zeros(len(F))
        conf = np.where(~has_cf & (votes < 500), "low", np.where(~has_cf & (votes < 1500) & (conf == "high"), "medium", conf))
        F["confidence"] = conf
        return F
