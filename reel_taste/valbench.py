"""Validation bench: judge model changes on time-based slices inside the training years.

The offline test set (the most recent 20% of ratings) is for the final check only. Every design
choice is made here instead: for each fold, fit on ratings dated before the fold's start and score the
ratings inside it. Both folds end before the test set starts, so the test set is never seen.

    from reel_taste import valbench
    base = valbench.run()                         # current model
    mine = valbench.run(model=MyHybrid)           # any class with fit(train) -> self, predict(df) -> F["pred"]
    valbench.compare(base, mine)

Run directly for the current model's numbers: .venv/bin/python -m reel_taste.valbench
"""
from __future__ import annotations

import time
from functools import lru_cache

import numpy as np
import pandas as pd

from . import pipeline
from .evaluate import baselines, features, metrics, rated_films, split
from .model import Hybrid

# Fold boundaries (rating_date). Fold 2 ends where the offline test set begins.
FOLD_STARTS = ["2022-09-01", "2023-09-01"]


@lru_cache(maxsize=1)
def data():
    """(ctx, train_years) — train_years is everything before the offline test set."""
    uf = pipeline.merged_user_films()
    ctx = pipeline.build_context()
    train, test = split(rated_films(uf))
    return ctx, train.reset_index(drop=True), str(test["rating_date"].min())


def folds():
    ctx, tr, test_from = data()
    ends = FOLD_STARTS[1:] + [test_from]
    for start, end in zip(FOLD_STARTS, ends):
        fit = tr[tr["rating_date"].fillna("") < start]
        val = tr[(tr["rating_date"] >= start) & (tr["rating_date"] < end)]
        yield f"{start}..{end}", fit.copy(), val.copy()


def run(model=Hybrid, label: str = "", baselines_too: bool = True, **kw) -> dict:
    """Fit `model(ctx, **kw)` on each fold and score it. Returns per-fold and pooled metrics, split
    into films with crowd data ("full") and without ("content"), with the TMDB crowd average beside."""
    ctx, _, _ = data()
    t0 = time.time()
    ys, ps, cs, crowd_ps, routes = [], [], [], [], []
    per_fold = {}
    for name, fit, val in folds():
        m = model(ctx, **kw).fit(fit)
        F = m.predict(val)
        p = F["pred"].to_numpy(float)
        y = val["rating"].to_numpy(float)
        has_cf = val["ml_id"].map(lambda x: pd.notna(x) and ctx.cf is not None and int(x) in ctx.cf.pos).to_numpy(bool)
        per_fold[name] = metrics(y, p)
        ys.append(y); ps.append(p); routes.append(has_cf)
        if baselines_too:
            Ftr = features(fit, fit, ctx)
            crowd_ps.append(baselines(fit, val, Ftr, F)["Crowd average (TMDB)"])
    y, p, r = np.concatenate(ys), np.concatenate(ps), np.concatenate(routes)
    out = dict(label=label or getattr(model, "__name__", "model"), seconds=round(time.time() - t0, 1),
               folds=per_fold, all=metrics(y, p), full=metrics(y[r], p[r]), content=metrics(y[~r], p[~r]))
    if baselines_too:
        c = np.concatenate(crowd_ps)
        out["crowd"] = dict(all=metrics(y, c), full=metrics(y[r], c[r]), content=metrics(y[~r], c[~r]))
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
    for row in ("all", "full", "content"):
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
