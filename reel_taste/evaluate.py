"""Offline test: hold out the most recent 20% of ratings by date, predict them, compare
against baselines. Reported as 'films with crowd data', 'new releases (content only)', 'all'."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

from .config import ARTIFACTS, HOLDOUT_FRACTION
from .letterboxd import evidence_weight, rewatches_before
from .model import Context, Hybrid, features

SCOREBOARD = ARTIFACTS / "scoreboard.json"


def item_key(r) -> str:
    if pd.notna(r["tmdb_id"]) and r["kind"] in ("movie", "tv"):
        return f"{r['kind']}:{int(r['tmdb_id'])}"
    return f"lb:{r['key']}"


def rated_films(uf: pd.DataFrame) -> pd.DataFrame:
    """Film training rows (shows are split out for the Shows tab)."""
    df = uf[uf["rating"].notna() & uf["kind"].ne("tv")].copy()
    df["key"] = df.apply(item_key, axis=1)
    df["w"] = evidence_weight(df).to_numpy()
    df = df.drop_duplicates("key")
    # Undated entries sort first (treated as old), so they can never be forced into the test set.
    return df.sort_values(["rating_date", "name"], na_position="first").reset_index(drop=True)


def split(df: pd.DataFrame):
    if len(df) < 20:
        raise ValueError(f"need at least 20 rated films for the offline test, have {len(df)}")
    n_test = max(1, int(round(len(df) * HOLDOUT_FRACTION)))
    return df.iloc[:-n_test].copy(), df.iloc[-n_test:].copy()


def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    out = dict(n=int(len(y)), rmse=float(np.sqrt(np.mean((p - y) ** 2))), mae=float(np.mean(np.abs(p - y))))
    out["spearman"] = float(spearmanr(y, p).statistic) if len(set(p)) > 1 else 0.0
    pos = y >= 4
    out["auc"] = float(roc_auc_score(pos, p)) if 0 < pos.sum() < len(y) and len(set(p)) > 1 else 0.5
    k = max(1, len(y) // 4)
    top = np.argsort(-p, kind="stable")[:k]
    out["hit_top25"] = float(pos[top].mean())
    out["base_rate"] = float(pos.mean())
    return out


def baselines(train: pd.DataFrame, test: pd.DataFrame, Ftr: pd.DataFrame, Fte: pd.DataFrame) -> dict:
    ubar = float(np.average(train["rating"], weights=train["w"]))
    y = train["rating"].to_numpy(float)
    out = {"Your average": np.full(len(test), ubar)}

    def calibrated(col):
        # Linear calibration onto your scale, fitted on your training ratings only.
        a, b = Ftr[col].to_numpy(float), Fte[col].to_numpy(float)
        m = ~np.isnan(a)
        if m.sum() < 10:
            return np.full(len(test), ubar)
        lr = LinearRegression().fit(a[m, None], y[m])
        return np.where(np.isnan(b), ubar, lr.predict(np.nan_to_num(b)[:, None]))

    out["Crowd average (TMDB)"] = calibrated("tmdb_raw")  # plain average, as a person would see it
    out["Popularity"] = calibrated("tmdb_logn")
    out["TMDB similar"] = np.where(Fte["link_n"].fillna(0) > 0, ubar + Fte["link"].fillna(0), ubar) \
        if "link" in Fte else np.full(len(test), ubar)
    out["Standard CF (MovieLens)"] = np.clip(Fte["cf_pred"].fillna(ubar).to_numpy(float), 0.5, 5)
    return out


def evaluate(ctx: Context, uf: pd.DataFrame, log=print) -> dict:
    df = rated_films(uf)
    train, test = split(df)
    # Training rows as they were when the test period began: only rewatches logged before it count.
    test_from = str(test["rating_date"].min())
    train = train.assign(rewatches=rewatches_before(train, test_from))
    train["w"] = evidence_weight(train).to_numpy()
    log(f"train {len(train)}  test {len(test)}  (test from {test['rating_date'].min()})")
    t0 = time.time()
    hy = Hybrid(ctx).fit(train)
    Fte = hy.predict(test)
    Ftr = features(train, train, ctx)  # in-sample, used only to calibrate crowd/popularity baselines
    preds = baselines(train, test, Ftr, Fte)
    preds["Hybrid (ours)"] = Fte["pred"].to_numpy()
    crowd = test["ml_id"].map(lambda m: pd.notna(m) and ctx.cf is not None and int(m) in ctx.cf.pos).to_numpy(bool)
    rows = {"Films with crowd data": crowd, "New releases (content only)": ~crowd, "All held-out films": np.ones(len(test), bool)}
    y = test["rating"].to_numpy(float)
    table = {}
    for row, m in rows.items():
        if m.sum() == 0:
            continue
        table[row] = {name: metrics(y[m], p[m]) for name, p in preds.items()}

    # Paired bootstrap: how often does ours beat the best baseline on RMSE?
    rng = np.random.default_rng(0)
    boot = {}
    for row, m in rows.items():
        if m.sum() < 5:
            continue
        base = {k: v for k, v in table[row].items() if k != "Hybrid (ours)"}
        best = min(base, key=lambda k: base[k]["rmse"])
        ix = np.where(m)[0]
        wins = 0
        for _ in range(2000):
            s = rng.choice(ix, len(ix))
            wins += np.mean((preds["Hybrid (ours)"][s] - y[s]) ** 2) < np.mean((preds[best][s] - y[s]) ** 2)
        boot[row] = dict(best_baseline=best, p_better=wins / 2000)

    res = dict(ts=time.strftime("%Y-%m-%d %H:%M"), n_train=len(train), n_test=len(test),
               test_from=str(test["rating_date"].min()), table=table, bootstrap=boot,
               learners=hy.describe(), cv=hy.cv,
               seconds=round(time.time() - t0, 1),
               examples=[dict(name=r.name, year=r.year, rating=r.rating, pred=round(p, 2), route=rt)
                         for r, p, rt in zip(test.itertuples(), Fte["pred"], Fte["route"])])
    res["watchlist"] = watchlist_test(ctx, uf, df, log)
    SCOREBOARD.write_text(json.dumps(res, indent=1, default=float))
    return res


def watchlist_test(ctx: Context, uf: pd.DataFrame, rated: pd.DataFrame, log=print) -> dict | None:
    """Train on everything rated; do films you flagged yourself rank high among unseen films?"""
    if not ctx.meta:
        return None
    wl = uf[uf["watchlist"] & ~uf["watched"] & uf["kind"].eq("movie") & uf["tmdb_id"].notna()]
    wl_keys = {f"movie:{int(t)}" for t in wl["tmdb_id"]}
    seen = set(rated["key"]) | {f"movie:{int(t)}" for t in uf.loc[uf["watched"] & uf["tmdb_id"].notna(), "tmdb_id"]}
    cands = [k for k, m in ctx.meta.items() if k.startswith("movie:") and k not in seen
             and ((m.get("vote_count") or 0) >= 50 or k in wl_keys)]
    T = pd.DataFrame({"key": cands})
    T["ml_id"] = pd.array([ctx.meta[k].get("ml_id") for k in cands], dtype="Int64")
    hy = Hybrid(ctx).fit(rated)
    F = hy.predict(T)
    is_wl = T["key"].isin(wl_keys).to_numpy()
    ubar = float(np.average(rated["rating"], weights=rated["w"]))
    scores = {
        "Hybrid (ours)": F["pred"].to_numpy(),
        "Standard CF (MovieLens)": F["cf_pred"].fillna(ubar).to_numpy(),
        "Crowd average (TMDB)": F["tmdb_raw"].fillna(0).to_numpy(),
        "Popularity": F["tmdb_logn"].fillna(0).to_numpy(),
        "TMDB similar": F["link"].fillna(0).to_numpy() + 0.01 * F["link_n"].fillna(0).to_numpy(),
    }
    out = {}
    for name, s in scores.items():
        order = np.argsort(-s, kind="stable")
        rank = np.empty(len(s), int)
        rank[order] = np.arange(len(s))
        r = rank[is_wl]
        out[name] = dict(median_percentile=float(np.median(1 - r / len(s)) * 100),
                         in_top_500=int((r < 500).sum()), in_top_2000=int((r < 2000).sum()))
    log(f"watchlist test: {is_wl.sum()} watchlist films among {len(s):,} candidates")
    return dict(n_watchlist=int(is_wl.sum()), n_candidates=int(len(T)), methods=out)
