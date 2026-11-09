"""'Your taste' profile and the live test."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from . import db, pipeline
from .evaluate import SCOREBOARD, item_key, rated_films

PREFIX = {"k_": "keyword", "d_": "director", "g_": "genre", "w_": "writer", "p_": "cinematographer",
          "m_": "composer", "c_": "actor", "l_": "language", "n_": "country", "y_": "decade"}


def taste() -> dict:
    uf = pipeline.merged_user_films()
    ctx = pipeline.context()
    rated = rated_films(uf)
    r = rated["rating"]
    out = dict(
        counts=dict(watched=int(uf["watched"].sum()), rated=int(uf["rating"].notna().sum()),
                    reviews=int((uf["review"].str.len() > 0).sum()), watchlist=int(uf["watchlist"].sum()),
                    rewatched=int((uf["rewatches"] > 0).sum()),
                    shows_rated=int(uf["kind"].eq("tv").sum() + db.show_ratings()["status"].eq("rated").sum())),
        mean=round(float(r.mean()), 2),
        share_4plus=round(float((r >= 4).mean()), 3),
        histogram={str(k): int(v) for k, v in r.value_counts().sort_index().items()},
    )
    d = rated[~rated["backfill"]].copy()
    d["yr"] = d["rating_date"].str[:4]
    out["by_year"] = [dict(year=y, n=int(len(g)), mean=round(float(g["rating"].mean()), 2),
                           share_4plus=round(float((g["rating"] >= 4).mean()), 3)) for y, g in d.groupby("yr")]
    out["backfill_n"] = int(rated["backfill"].sum())
    rated["decade"] = (rated["year"] // 10 * 10).astype("Int64")
    out["by_decade"] = [dict(decade=int(k), n=int(len(g)), mean=round(float(g["rating"].mean()), 2))
                        for k, g in rated.dropna(subset=["decade"]).groupby("decade")]
    since = rated[rated["rating_date"] >= "2023"]
    out["live_bar"] = round(float((since["rating"] >= 4).mean()), 3) if len(since) else None

    meta = ctx.meta
    if meta:
        rows = []
        for x in rated.itertuples():
            m = meta.get(x.key)
            if m:
                rows.append((x, m))
        dirs, gens, langs = {}, {}, {}
        for x, m in rows:
            for dd in (m.get("directors") or [])[:2]:
                dirs.setdefault(dd, []).append(x.rating)
            for g in m.get("genres") or []:
                gens.setdefault(g, []).append(x.rating)
            langs.setdefault(m.get("language") or "?", []).append(x.rating)
        out["directors"] = sorted([dict(name=k, n=len(v), mean=round(float(np.mean(v)), 2)) for k, v in dirs.items() if len(v) >= 3],
                                  key=lambda z: (-z["n"], -z["mean"]))[:15]
        out["genres"] = sorted([dict(name=k, n=len(v), mean=round(float(np.mean(v)), 2)) for k, v in gens.items() if len(v) >= 5],
                               key=lambda z: -z["mean"])
        out["languages"] = sorted([dict(name=k, n=len(v), mean=round(float(np.mean(v)), 2)) for k, v in langs.items()],
                                  key=lambda z: -z["n"])[:8]
        gaps = []
        for x, m in rows:
            if (m.get("vote_count") or 0) >= 200 and m.get("vote_average"):
                crowd = m["vote_average"] / 2
                gaps.append(dict(name=x.name, year=None if pd.isna(x.year) else int(x.year),
                                 yours=x.rating, crowd=round(crowd, 1), gap=round(x.rating - crowd, 2)))
        gaps.sort(key=lambda z: -z["gap"])
        out["above_crowd"] = gaps[:12]
        out["below_crowd"] = gaps[::-1][:12]
        out["crowd_gap_mean"] = round(float(np.mean([g["gap"] for g in gaps])), 2) if gaps else None
        if ctx.cs is not None:
            out["fingerprint"] = fingerprint(rated, ctx)
    return out


def fingerprint(rated: pd.DataFrame, ctx) -> dict:
    cs = ctx.cs
    rows = np.array([cs.index.get(k, -1) for k in rated["key"]])
    ok = rows >= 0
    y = rated["rating"].to_numpy(float)[ok]
    X = cs.X[rows[ok]]
    n_tags = len(cs.tags.get_feature_names_out())
    X = X[:, :n_tags]
    df = np.asarray((X > 0).sum(0)).ravel()
    m = Ridge(alpha=3.0).fit(X, y - y.mean())
    coef = m.coef_
    names = cs.vocab[:n_tags]
    res = {"loves": [], "avoids": []}
    for i in np.argsort(-coef):
        if df[i] >= 4 and names[i][:2] in PREFIX and len(res["loves"]) < 18:
            res["loves"].append(dict(kind=PREFIX[names[i][:2]], name=names[i][2:].replace("_", " "), n=int(df[i])))
    for i in np.argsort(coef):
        if df[i] >= 4 and names[i][:2] in PREFIX and len(res["avoids"]) < 18:
            res["avoids"].append(dict(kind=PREFIX[names[i][:2]], name=names[i][2:].replace("_", " "), n=int(df[i])))
    return res


def live_test() -> dict:
    log = db.q("SELECT * FROM rec_log")
    uf = pipeline.merged_user_films()
    since = uf[uf["rating"].notna() & ~uf["backfill"] & (uf["rating_date"] >= "2023")]
    bar = float((since["rating"] >= 4).mean()) if len(since) else None
    if len(log) == 0:
        return dict(bar=bar, sources={}, n_queries=0)
    later = {}
    for r in uf[uf["rating"].notna() & uf["tmdb_id"].notna()].itertuples():
        later[(r.kind or "movie", int(r.tmdb_id))] = (r.rating, str(r.rating_date))
    for r in db.show_ratings().query("status == 'rated'").itertuples():
        later[("tv", int(r.tmdb_id))] = (r.rating, time.strftime("%Y-%m-%d", time.localtime(r.ts)))
    out = {}
    first = log.sort_values("ts").drop_duplicates(["source", "kind", "tmdb_id"])
    for (src, kind), g in first.groupby(["source", "kind"]):
        watched = []
        for r in g.itertuples():
            hit = later.get((kind, int(r.tmdb_id)))
            if hit and hit[1] >= time.strftime("%Y-%m-%d", time.localtime(r.ts)):
                watched.append(hit[0])
        out[f"{kind}:{src}"] = dict(kind=kind, source=src, recommended=int(len(g)), watched=len(watched),
                                    hits=int(sum(w >= 4 for w in watched)),
                                    hit_rate=round(sum(w >= 4 for w in watched) / len(watched), 3) if watched else None,
                                    mean_rating=round(float(np.mean(watched)), 2) if watched else None)
    fb = db.feedback()
    return dict(bar=bar, sources=out, n_queries=int(log["query_id"].nunique()),
                feedback={a: int(n) for a, n in fb["action"].value_counts().items()} if len(fb) else {})


def scoreboard() -> dict:
    off = json.loads(SCOREBOARD.read_text()) if SCOREBOARD.exists() else None
    return dict(offline=off, live=live_test())
