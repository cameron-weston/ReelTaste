"""Validation bench: judge model changes on time-based slices inside the training years.

The offline test set (the most recent 20% of ratings) is for the final check only. Every design
choice is made here instead: for each fold, fit on ratings dated before the fold's start and score the
ratings inside it. Both folds end before the test set starts, so the test set is never seen.

Each fold is rebuilt as it would have looked on its start date:
- training weights count only rewatches logged before the start date;
- a film counts as having MovieLens crowd data only if it had enough MovieLens ratings before the
  start date (MovieLens runs to Oct 2023, so without this, 2022 releases look well known in 2022).
One thing can't be rewound: TMDB vote averages and counts are today's snapshot.

    from reel_taste import valbench
    base = valbench.run()                         # current model
    mine = valbench.run(model=MyModel)            # any class with fit(train) -> self, predict(df) -> F["pred"]
    valbench.compare(base, mine)

Run directly for the current model's numbers: .venv/bin/python -m reel_taste.valbench
"""
from __future__ import annotations

import time
from functools import lru_cache

import numpy as np
import pandas as pd

from . import letterboxd, pipeline
from .cf import MIN_ITEM_RATINGS
from .config import ARTIFACTS, ML_DIR
from .evaluate import baselines, features, metrics, rated_films, split
from .model import Hybrid

# Fold boundaries (rating_date). The last fold ends where the offline test set begins.
FOLD_STARTS = ["2022-09-01", "2023-09-01"]
ML_COUNTS = ARTIFACTS / "ml_counts_at_folds.parquet"


@lru_cache(maxsize=1)
def data():
    """(ctx, train_years, test_from) — train_years is everything before the offline test set."""
    uf = pipeline.merged_user_films()
    ctx = pipeline.build_context()
    train, test = split(rated_films(uf))
    return ctx, train.reset_index(drop=True), str(test["rating_date"].min())


@lru_cache(maxsize=1)
def ml_counts() -> pd.DataFrame:
    """MovieLens ratings per film before each fold start (cached; built once from ratings.csv)."""
    if ML_COUNTS.exists():
        c = pd.read_parquet(ML_COUNTS)
        if all(s in c.columns for s in FOLD_STARTS):
            return c
    r = pd.read_csv(ML_DIR / "ratings.csv", usecols=["movieId", "timestamp"],
                    dtype={"movieId": "int32", "timestamp": "int64"})
    out = {}
    for s in FOLD_STARTS:
        cut = int(pd.Timestamp(s).timestamp())
        out[s] = r.loc[r["timestamp"] < cut, "movieId"].value_counts()
    c = pd.DataFrame(out).fillna(0).astype("int32")
    c.index.name = "ml_id"
    c.to_parquet(ML_COUNTS)
    return c


def as_of(df: pd.DataFrame, start: str, fit: bool) -> pd.DataFrame:
    """The rows as they'd have looked on `start`: crowd data and (for training rows) weights."""
    df = df.copy()
    cnt = ml_counts()[start]
    known = df["ml_id"].map(lambda m: pd.notna(m) and int(cnt.get(int(m), 0)) >= MIN_ITEM_RATINGS)
    df["ml_id"] = df["ml_id"].where(known.to_numpy(bool), pd.NA).astype("Int64")
    df["rewatches"] = letterboxd.rewatches_before(df, start if fit else "9999")
    if fit:
        df["w"] = letterboxd.evidence_weight(df).to_numpy()
    return df


def folds():
    ctx, tr, test_from = data()
    ends = FOLD_STARTS[1:] + [test_from]
    for start, end in zip(FOLD_STARTS, ends):
        fit = tr[tr["rating_date"].fillna("") < start]
        val = tr[(tr["rating_date"] >= start) & (tr["rating_date"] < end)]
        yield f"{start}..{end}", as_of(fit, start, True), as_of(val, start, False)


def run(model=Hybrid, label: str = "", baselines_too: bool = True, **kw) -> dict:
    """Fit `model(ctx, **kw)` on each fold and score it. Returns per-fold and pooled metrics, split
    into films with crowd data ("full") and without ("content"), plus first watches only ("first",
    what recommendations are actually for), with the TMDB crowd average beside."""
    ctx, _, _ = data()
    t0 = time.time()
    ys, ps, routes, firsts, crowd_ps = [], [], [], [], []
    per_fold = {}
    for name, fit, val in folds():
        m = model(ctx, **kw).fit(fit)
        F = m.predict(val)
        p = F["pred"].to_numpy(float)
        y = val["rating"].to_numpy(float)
        per_fold[name] = metrics(y, p)
        ys.append(y); ps.append(p)
        routes.append(val["ml_id"].notna().to_numpy(bool))
        firsts.append((val["rewatches"] == 0).to_numpy(bool))  # never rewatched: rated on first watch
        if baselines_too:
            crowd_ps.append(baselines(fit, val, features(fit, fit, ctx), F)["Crowd average (TMDB)"])
    y, p, r, f = (np.concatenate(x) for x in (ys, ps, routes, firsts))
    rows = dict(all=np.ones(len(y), bool), full=r, content=~r, first=f)
    out = dict(label=label or getattr(model, "__name__", "model"), seconds=round(time.time() - t0, 1),
               folds=per_fold, **{k: metrics(y[m], p[m]) for k, m in rows.items()})
    if baselines_too:
        c = np.concatenate(crowd_ps)
        out["crowd"] = {k: metrics(y[m], c[m]) for k, m in rows.items()}
    out["_y"], out["_p"] = y, p  # for paired comparisons
    return out


def compare(a: dict, b: dict, n_boot: int = 2000) -> dict:
    """Paired bootstrap of b vs a on pooled squared error: share of resamples where b is better."""
    y, pa, pb = a["_y"], a["_p"], b["_p"]
    rng = np.random.default_rng(0)
    wins = 0
    for _ in range(n_boot):
        s = rng.integers(0, len(y), len(y))
        wins += np.mean((pb[s] - y[s]) ** 2) < np.mean((pa[s] - y[s]) ** 2)
    return dict(p_b_better=wins / n_boot, rmse_a=a["all"]["rmse"], rmse_b=b["all"]["rmse"],
                spearman_a=a["all"]["spearman"], spearman_b=b["all"]["spearman"],
                auc_a=a["all"]["auc"], auc_b=b["all"]["auc"])


def show(res: dict) -> str:
    keys = ("rmse", "mae", "spearman", "auc", "hit_top25")
    lines = [f"{res['label']}  ({res['seconds']}s)"]
    for row in ("all", "full", "content", "first"):
        m = res[row]
        s = f"  {row:8s} n={m['n']:4d} " + " ".join(f"{k}={m[k]:.3f}" for k in keys)
        if "crowd" in res:
            c = res["crowd"][row]
            s += "   | crowd " + " ".join(f"{k}={c[k]:.3f}" for k in keys)
        lines.append(s)
    for f, m in res["folds"].items():
        lines.append(f"  fold {f}: n={m['n']} rmse={m['rmse']:.3f} spearman={m['spearman']:.3f}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(show(run(label="current Hybrid")))
