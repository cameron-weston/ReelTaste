"""Turn precomputed scores into a list: filters, obscurity dial, mood, diversity, why, logging."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import db, llm, pipeline
from . import mood as moodlib
from .tmdb import IMG

STAR = {0.5: "½", 1: "★", 1.5: "★½", 2: "★★", 2.5: "★★½", 3: "★★★", 3.5: "★★★½", 4: "★★★★", 4.5: "★★★★½", 5: "★★★★★"}


def _blocked(meta: dict, blocks: set[str], key: str) -> bool:
    if key in blocks:
        return True
    for g in meta.get("genres") or []:
        if f"genre:{g}" in blocks:
            return True
    for p in list(meta.get("directors") or []) + list(meta.get("cast") or [])[:5]:
        if f"person:{p}" in blocks:
            return True
    for k in meta.get("keywords") or []:
        if f"keyword:{k}" in blocks:
            return True
    return False


def _eligible(kind: str, m: dict, f: dict) -> bool:
    rt = m.get("runtime")
    rt = 0 if rt is None or pd.isna(rt) else rt
    if kind == "movie":
        if rt and rt < 60:
            return False
        if not f.get("include_docs", True) and "Documentary" in (m.get("genres") or []):
            return False
    if (m.get("vote_count") or 0) < f.get("min_votes", 200 if kind == "movie" else 300):
        return False
    if f.get("max_runtime") and kind == "movie" and (not rt or rt > f["max_runtime"]):
        return False
    y = m.get("year")
    y = None if y is None or pd.isna(y) else int(y)
    if f.get("year_min") and (y is None or y < f["year_min"]):
        return False
    if f.get("year_max") and (y is None or y > f["year_max"]):
        return False
    if f.get("languages") and m.get("language") not in f["languages"]:
        return False
    if kind == "tv":
        if f.get("finished_only") and m.get("status") not in ("Ended", "Canceled"):
            return False
        if f.get("max_seasons") and (m.get("seasons") or 99) > f["max_seasons"]:
            return False
    return True


def _neighbours(key: str, rated: pd.DataFrame, ctx, k: int = 3) -> list[dict]:
    """The user's own films most like this one, preferring ones they rated well."""
    cs, cf, meta = ctx.cs, ctx.cf, ctx.meta
    sims = np.zeros(len(rated))
    if cs is not None and key in cs.index:
        rows = np.array([cs.index.get(x, -1) for x in rated["key"]])
        ok = rows >= 0
        sims[ok] = cs.Z[rows[ok]] @ cs.Z[cs.index[key]]
    m = meta.get(key, {})
    v = cf.vec(m.get("ml_id")) if cf is not None else None
    if v is not None:
        cfs = np.array([(lambda u: float(u @ v) if u is not None else 0.0)(cf.vec(x)) for x in rated["ml_id"]])
        sims = np.maximum(sims, cfs)
    # Cite the closest films the user liked (3.5★+); closeness first, so broad favourites
    # don't show up as the "reason" for everything.
    liked = rated["rating"].to_numpy() >= 3.5
    score = np.where(liked, sims, -1.0)
    out = []
    for i in np.argsort(-score)[:k]:
        if score[i] <= 0.15:
            break
        r = rated.iloc[i]
        out.append(dict(name=r["name"], year=None if pd.isna(r["year"]) else int(r["year"]),
                        rating=float(r["rating"]), sim=round(float(sims[i]), 2), key=r["key"]))
    return out


def _shared(key: str, nbrs: list[dict], meta: dict) -> list[str]:
    m = meta.get(key, {})
    out, people = [], set()
    for role, field in (("director", "directors"), ("writer", "writers"), ("cinematographer", "dop"), ("composer", "composer")):
        mine = set(m.get(field) or [])
        for n in nbrs:
            for p in sorted(mine & set(meta.get(n["key"], {}).get(field) or [])):
                if p not in people:  # one mention per person, at their most notable role
                    people.add(p)
                    out.append(f"same {role} ({p})")
    kws = {}
    for n in nbrs:
        for kw in set(m.get("keywords") or []) & set(meta.get(n["key"], {}).get("keywords") or []):
            kws[kw] = kws.get(kw, 0) + 1
    top = [k for k, _ in sorted(kws.items(), key=lambda x: -x[1])][:3]
    if top:
        out.append("shares: " + ", ".join(top))
    return out[:3]


def _why_text(nbrs, shared) -> str:
    if not nbrs:
        return "Fits your overall taste profile; no close match among films you've rated."
    names = " and ".join(f"{n['name']} ({STAR.get(n['rating'], n['rating'])})" for n in nbrs[:2])
    s = f"Close to {names}."
    if shared:
        txt = "; ".join(shared)
        s += " " + txt[0].upper() + txt[1:] + "."
    return s


def _card(key: str, m: dict, row, ctx, rated) -> dict:
    kind = key.split(":")[0]
    nbrs = _neighbours(key, rated, ctx)
    shared = _shared(key, nbrs, ctx.meta)
    y = m.get("year")
    card = dict(key=key, kind=kind, tmdb_id=int(m["id"]), title=m.get("title"),
                year=None if y is None or pd.isna(y) else int(y),
                poster=f"{IMG}{m['poster']}" if m.get("poster") else None,
                overview=m.get("overview"), genres=list(m.get("genres") or []),
                directors=list(m.get("directors") or []), runtime=m.get("runtime"),
                language=m.get("language"), vote_average=m.get("vote_average"), vote_count=m.get("vote_count"),
                predicted=round(float(row["pred"]), 2), confidence=row["confidence"],
                because=nbrs, shared=shared, why=_why_text(nbrs, shared),
                tmdb_url=f"https://www.themoviedb.org/{kind}/{m['id']}",
                letterboxd_url=f"https://letterboxd.com/tmdb/{m['id']}" if kind == "movie" else None)
    if kind == "tv":
        card.update(seasons=m.get("seasons"), episodes=m.get("episodes"), status=m.get("status"),
                    networks=list(m.get("networks") or []))
    return card


def _diverse(order: list[str], meta: dict, Z, index, n: int) -> list[str]:
    picked, dirs = [], {}
    for k in order:
        m = meta[k]
        ds = list(m.get("directors") or [])[:1]
        if any(dirs.get(d, 0) >= 2 for d in ds):
            continue
        if Z is not None and k in index and picked:
            pv = [index[p] for p in picked if p in index]
            if pv and (Z[pv] @ Z[index[k]] > 0.8).sum() >= 2:  # same narrow niche
                continue
        picked.append(k)
        for d in ds:
            dirs[d] = dirs.get(d, 0) + 1
        if len(picked) >= n:
            break
    return picked


def _taste_brief(rated: pd.DataFrame, n_top: int = 40) -> str:
    r = rated.copy()
    r["recency"] = pd.to_datetime(r["rating_date"], errors="coerce")
    top = r.sort_values(["rating", "recency"], ascending=False).head(n_top)
    low = r.sort_values(["rating", "recency"], ascending=[True, False]).head(15)

    def line(x):
        rev = (x.review or "").replace("\n", " ")[:140]
        return f"- {x.name} ({'' if pd.isna(x.year) else int(x.year)}): {x.rating}★" + (f' — "{rev}"' if rev else "")
    return ("Films they rated highest:\n" + "\n".join(line(x) for x in top.itertuples()) +
            "\n\nFilms they rated lowest:\n" + "\n".join(line(x) for x in low.itertuples()))


def _llm_rerank(kind: str, mood: str, cands: list[dict], rated: pd.DataFrame, n: int):
    noun = "film" if kind == "movie" else "TV show"
    lines = []
    for c in cands:
        extra = f"{c['seasons']} seasons, {c['status']}" if kind == "tv" else f"{c['runtime']} min"
        lines.append(f"[{c['tmdb_id']}] {c['title']} ({c['year']}) — {', '.join(c['directors'][:2])} — "
                     f"{', '.join(c['genres'][:3])} — {extra} — predicted {c['predicted']}★ — "
                     f"{(c['overview'] or '')[:220]}")
    prompt = f"""You are the final stage of a personal {noun} recommender for one viewer.
A model has already shortlisted these candidates and predicted the viewer's rating for each.
Your job: pick the {n} best for the viewer's request, in order, and say why in one or two sentences
that cite specific films the viewer rated (from the taste profile below). Respect the request strictly
(mood, length, tone). Prefer high predicted ratings, but the request matters more. Don't pick two that
are near-duplicates in feel. Never invent films outside the list.

Viewer's request: "{mood or 'no particular mood — just the best picks for them'}"

{_taste_brief(rated)}

Candidates:
{chr(10).join(lines)}

Answer with only a JSON array of objects: [{{"id": <tmdb id>, "why": "<one or two sentences>"}}]"""
    out = llm.parse_json(llm.complete(prompt, timeout=150))
    by_id = {c["tmdb_id"]: c for c in cands}
    picked, ids = [], set()
    for o in out if isinstance(out, list) else []:
        try:
            i = int(o.get("id", -1))
        except (TypeError, ValueError):
            continue
        c = by_id.get(i)
        if c is not None and i not in ids:
            picked.append(dict(c, why=str(o.get("why") or c["why"]), why_source="llm"))
            ids.add(i)
        if len(picked) >= n:
            break
    return picked


def recommend(kind: str = "movie", mode: str = "discover", mood: str = "", filters: dict | None = None,
              dial: float = 0.5, n: int = 10, offset: int = 0, use_llm: bool = True,
              query_id: str | None = None, exclude: list[str] | None = None) -> dict:
    filters = filters or {}
    state = pipeline.load_state()
    table = state and state.get("films" if kind == "movie" else "shows")
    if table is None:
        return dict(items=[], error="Not built yet. Add the TMDB key and run the build.")
    ctx = pipeline.context()
    meta = ctx.meta
    blocks = set(db.blocklist()["value"])
    fb = db.feedback()
    dismissed = {f"{r.kind}:{r.tmdb_id}" for r in fb[fb["action"].isin(["not_interested", "watched"])].itertuples()}
    seen, wl = state["seen"], state["watchlist"]
    parsed = moodlib.parse(mood) if mood else {"filters": {}, "want": [], "avoid": []}
    if kind == "tv":
        parsed["filters"].pop("max_runtime", None)  # episode length isn't what "under two hours" means
    filters = {**parsed["filters"], **filters}  # explicit filter controls win over the mood text
    allow_standup = "stand" in (mood or "").lower()

    keys = []
    for k in table.index:
        m = meta.get(k)
        if m is None or k in seen or k in dismissed or _blocked(m, blocks, k):
            continue
        if not allow_standup and any("stand-up" in x or "stand up" in x for x in (m.get("keywords") or [])):
            continue
        if kind == "movie" and mode == "watchlist" and k not in wl:
            continue
        if kind == "movie" and mode == "discover" and k in wl:
            continue
        f = filters if mode != "watchlist" else {**filters, "min_votes": 0}
        if not _eligible(kind, m, f):
            continue
        keys.append(k)
    if not keys:
        return dict(items=[], query_id=None, note="Nothing matches those filters.")
    T = table.loc[keys]
    pop = np.log1p(np.array([meta[k].get("vote_count") or 0 for k in keys], float))
    popz = (pop - pop.mean()) / (pop.std() + 1e-9)
    # Obscurity dial: 0 = no popularity penalty, 0.5 ≈ 0.2★ per SD of log-votes, 1 = 0.6★ per SD.
    score = T["pred"].to_numpy() - 0.6 * float(dial) ** 1.5 * popz * (mode != "watchlist")
    # Rank by a cautious estimate: thinly-supported predictions need to clear a higher bar.
    score = score - T["confidence"].map({"low": 0.3, "medium": 0.1, "high": 0.0}).to_numpy()
    # Titles without MovieLens data lean on the TMDB average, which is fan-inflated at low vote
    # counts. Their ranking pays for that uncertainty; predictions themselves are unchanged.
    votes = np.array([meta[k].get("vote_count") or 0 for k in keys], float)
    content_only = (T["route"].to_numpy() == "content")
    if kind == "tv":  # every show is content-only and TV fan bases inflate small samples harder
        score = score - 0.9 * np.sqrt(1000 / (votes + 1000))
    else:
        score = score - content_only * 0.6 * np.sqrt(500 / (votes + 500))
    fit = np.zeros(len(keys))
    if mood:
        # Mood fit: lexicon (genres/keywords, with negations) plus plot-text similarity.
        lex = np.array([moodlib.score(meta[k], parsed) for k in keys])
        fit = 1.5 * lex
        if ctx.cs is not None:
            mv = ctx.cs.embed_text(mood)
            rows = [ctx.cs.index.get(k) for k in keys]
            sim = np.array([float(ctx.cs.Z[r] @ mv) if r is not None else 0.0 for r in rows])
            fit = fit + 0.25 * (sim - sim.mean()) / (sim.std() + 1e-9)
    order = [keys[i] for i in np.argsort(-(score + fit))]
    # Claude's shortlist leans harder on the mood: it can judge fit, but only among what it's given.
    llm_order = [keys[i] for i in np.argsort(-(score + 2.0 * fit))] if mood else order
    rated = state["rated"] if kind == "movie" else state.get("tv_rated", state["rated"])
    # "More" pages past everything already on screen (Claude's picks included), not a fixed slice.
    shown = set(exclude or [])
    want = len(shown) + n

    llm_used, llm_error = False, None
    Z = ctx.cs.Z if ctx.cs is not None else None
    idx = ctx.cs.index if ctx.cs is not None else {}
    pool = _diverse(order, meta, Z, idx, want)
    if use_llm and offset == 0:
        cands = [_card(k, meta[k], T.loc[k], ctx, rated) for k in _diverse(llm_order, meta, Z, idx, 50)]
        try:
            items = _llm_rerank(kind, mood, cands, rated, n)
            llm_used = len(items) > 0
            if len(items) < n:
                got = {c["tmdb_id"] for c in items}
                items += [c for c in cands if c["tmdb_id"] not in got][: n - len(items)]
        except ValueError:
            llm_error = "Claude's reply wasn't usable JSON"
            items = None
        except Exception as e:  # noqa: BLE001
            llm_error = str(e)[:200]
            items = None
        if items is None:  # fall back to the model's own order
            items = [_card(k, meta[k], T.loc[k], ctx, rated) for k in pool[:n]]
    else:
        items = [_card(k, meta[k], T.loc[k], ctx, rated) for k in pool if k not in shown][:n]

    qid = query_id or db.new_query(kind, mode, mood, filters, dial, llm_used)
    db.log_recs(qid, kind, "llm" if llm_used else "model", [(c["tmdb_id"], c["predicted"]) for c in items], True,
                start=len(shown))
    if offset == 0:
        _log_baselines(qid, kind, keys, T, meta, n)
    return dict(items=items, query_id=qid, llm_used=llm_used, llm_error=llm_error,
                n_candidates=len(keys), mood_applied=bool(mood))


def _log_baselines(qid, kind, keys, T, meta, n):
    """Silently log what the baselines would have shown, for the live comparison."""
    ubar = 0.0
    lists = {
        "crowd": sorted(keys, key=lambda k: -(meta[k].get("vote_average") or 0) if (meta[k].get("vote_count") or 0) >= 500 else 0),
        "popular": sorted(keys, key=lambda k: -(meta[k].get("vote_count") or 0)),
    }
    if kind == "movie":
        cfp = T["cf_pred"].fillna(ubar)
        lists["svd"] = list(cfp.sort_values(ascending=False).index)
    for src, order in lists.items():
        db.log_recs(qid, kind, src, [(meta[k]["id"], float("nan")) for k in order[:n]], False)
