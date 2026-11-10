"""End-to-end build: match → catalogs → content space → offline test → production scores."""
from __future__ import annotations

import os
import pickle
import tempfile
import threading
import time

import numpy as np
import pandas as pd

from . import catalog, db, letterboxd, movielens, tmdb
from .cf import CF
from .config import ARTIFACTS, BACKFILL_DATE
from .content import ContentSpace
from .evaluate import evaluate, item_key, rated_films
from .model import Context, Hybrid

STATE = ARTIFACTS / "state.pkl"
_lock = threading.Lock()


def build_context() -> Context:
    cf = CF.load()
    cs, meta = None, {}
    if catalog.FILM_CATALOG.exists():
        fc = catalog.film_catalog()
        frames = {"movie": fc}
        if catalog.TV_CATALOG.exists():
            frames["tv"] = catalog.tv_catalog()
        cs = ContentSpace.load() if (ARTIFACTS / "content.pkl").exists() else None
        for kind, df in frames.items():
            for r in df.to_dict("records"):
                for f, v in r.items():
                    if isinstance(v, np.ndarray):
                        r[f] = v.tolist()
                r["ml_id"] = r.get("ml_id") if kind == "movie" and pd.notna(r.get("ml_id")) else None
                meta[f"{kind}:{r['id']}"] = r
    return Context(cf=cf, cs=cs, meta=meta)


def merged_user_films() -> pd.DataFrame:
    """Export + newer RSS entries + in-app logs (ratings, watches, rewatches) merged into one table."""
    uf = catalog.user_films()
    rss = db.rss_entries().assign(source="rss", review="")
    # Editing a rating on Letterboxd yields a second RSS entry for the same watch; keep the newest.
    rss = rss.sort_values("ts").drop_duplicates(["name", "year", "watched_date"], keep="last")
    logs = db.app_logs()
    if len(logs):
        is_tv = logs["kind"].eq("tv")
        logs = logs.assign(source="app", tmdb_movie=logs["tmdb_id"].where(~is_tv),
                           tmdb_tv=logs["tmdb_id"].where(is_tv))
    cols = ["name", "year", "rating", "watched_date", "rewatch", "liked", "tmdb_movie", "tmdb_tv", "source",
            "review", "ts"]
    entries = [x[cols] for x in (rss, logs) if len(x)]
    if not entries:
        return uf
    entries = pd.concat(entries, ignore_index=True)
    entries["watched_date"] = entries["watched_date"].fillna(
        pd.to_datetime(entries["ts"], unit="s").dt.strftime("%Y-%m-%d"))
    entries = entries.sort_values(["watched_date", "ts"], kind="mergesort")

    t2m = movielens.tmdb_to_ml()
    # first_rating: the rating from a first watch (the export's rating stands in for its own rows).
    rows = uf.assign(first_rating=uf["rating"], first_rating_date=uf["rating_date"]).to_dict("records")
    n_export = len(rows)
    by_tmdb = {(r["kind"], int(r["tmdb_id"])): i for i, r in enumerate(rows) if pd.notna(r["tmdb_id"])}
    by_key = {r["key"]: i for i, r in enumerate(rows)}
    for r in entries.itertuples():
        kind, tid = ("tv", r.tmdb_tv) if pd.notna(r.tmdb_tv) else ("movie", r.tmdb_movie)
        tid = int(tid) if pd.notna(tid) else None
        key = letterboxd.key(r.name, r.year)
        i = by_tmdb.get((kind, tid)) if tid is not None else None
        if i is None:
            i = by_key.get(key)
        review = r.review if isinstance(r.review, str) else ""
        if i is None:
            rows.append(dict(key=key, name=r.name, year=int(r.year) if str(r.year).isdigit() else None,
                             rating=r.rating, rating_date=r.watched_date, watched=True,
                             rewatches=int(r.rewatch), liked=bool(r.liked), watchlist=False, review=review,
                             last_watched=r.watched_date, source=r.source, backfill=False, kind=kind,
                             tmdb_id=tid, ml_id=t2m.get(tid) if (tid is not None and kind == "movie") else None,
                             first_rating=None if r.rewatch else r.rating,
                             first_rating_date=None if (r.rewatch or pd.isna(r.rating)) else r.watched_date))
            i = len(rows) - 1
            by_key[key] = i
            if tid is not None:
                by_tmdb[(kind, tid)] = i
            continue
        row = rows[i]
        # The export already contains RSS entries up to its own last watch; don't count them twice.
        if r.source == "rss" and i < n_export and str(r.watched_date) <= str(row.get("last_watched") or ""):
            continue
        row["watched"] = True
        if pd.notna(r.rating) and (pd.isna(row["rating"]) or str(row["rating_date"]) <= str(r.watched_date)):
            row.update(rating=r.rating, rating_date=r.watched_date, backfill=False)
        if pd.notna(r.rating) and not r.rewatch and pd.isna(row.get("first_rating")):
            row.update(first_rating=r.rating, first_rating_date=r.watched_date)
        if r.rewatch:
            row["rewatches"] = int(row["rewatches"] or 0) + 1
        if r.liked:
            row["liked"] = True
        if review:
            row["review"] = review
        if str(r.watched_date) > str(row.get("last_watched") or ""):
            row["last_watched"] = r.watched_date
    out = pd.DataFrame(rows, columns=[*uf.columns, "first_rating", "first_rating_date"])
    out["first_rating"] = pd.to_numeric(out["first_rating"], errors="coerce")
    for c in ("tmdb_id", "ml_id"):
        out[c] = pd.array([int(v) if pd.notna(v) else None for v in out[c]], dtype="Int64")
    for c in ("watched", "liked", "backfill", "watchlist"):
        if c in out:
            out[c] = out[c].fillna(False).astype(bool)
    return out


def n_ratings() -> int:
    """Ratings the model learns from: films and shows in the merged table plus the show rating session."""
    sr = db.show_ratings()
    return int(merged_user_films()["rating"].notna().sum()) + int(sr["status"].eq("rated").sum())


def refresh_rss(log=print) -> int:
    try:
        n = db.save_rss(letterboxd.fetch_rss())
        db.set_meta("rss_last_fetch", time.strftime("%Y-%m-%d %H:%M"))
        return n
    except Exception as e:  # noqa: BLE001
        log(f"RSS fetch failed: {e}")
        return 0


def build(full: bool = True, log=print):
    """full=True rebuilds catalogs and the content space; False just retrains on new ratings."""
    with _lock:
        t0 = time.time()
        if full or not catalog.USER_FILMS.exists():
            catalog.match_user(log)
        if full and tmdb.has_key():
            catalog.build_film_catalog(log)
            catalog.build_tv_catalog(log)
            log("fitting content space")
            ContentSpace().fit({"movie": catalog.film_catalog(), "tv": catalog.tv_catalog()}).save()
        uf = merged_user_films()
        # Count what this build trains on now, so ratings logged while it runs still count as new.
        n_trained = int(uf["rating"].notna().sum()) + int(db.show_ratings()["status"].eq("rated").sum())
        ctx = build_context()
        log("offline test")
        evaluate(ctx, uf, log)
        log("scoring catalogs")
        state = score_all(ctx, uf, log)
        fd, tmp = tempfile.mkstemp(dir=STATE.parent, suffix=".tmp")  # unique, so two builds never share it
        with os.fdopen(fd, "wb") as f:
            pickle.dump(state, f)
        os.replace(tmp, STATE)  # readers never see a half-written file
        db.set_meta("trained_at", time.strftime("%Y-%m-%d %H:%M"))
        db.set_meta("trained_n_ratings", n_trained)
        log(f"build done in {time.time() - t0:.0f}s")
        return state


def _show_training(uf: pd.DataFrame) -> pd.DataFrame:
    """Show ratings: Letterboxd-logged shows plus ratings from the in-app session."""
    rows = []
    for r in uf[uf["kind"].eq("tv") & uf["rating"].notna()].itertuples():
        rows.append(dict(key=f"tv:{int(r.tmdb_id)}", rating=r.rating, rating_date=r.rating_date,
                         rewatches=r.rewatches, backfill=r.backfill))
    sr = db.show_ratings()
    for r in sr[sr["status"].eq("rated")].itertuples():
        # Rated in the "shows you've seen" session: past watches of unknown date, so they share the
        # backfill weight instead of counting as fresh evidence.
        rows.append(dict(key=f"tv:{int(r.tmdb_id)}", rating=r.rating,
                         rating_date=time.strftime("%Y-%m-%d", time.localtime(r.ts)), rewatches=0, backfill=True))
    df = pd.DataFrame(rows, columns=["key", "rating", "rating_date", "rewatches", "backfill"])
    if len(df):
        df = df.drop_duplicates("key", keep="last")
        df["w"] = letterboxd.evidence_weight(df).to_numpy() * 1.5  # direct TV evidence counts extra
        df["ml_id"] = pd.array([None] * len(df), dtype="Int64")
    return df


def score_all(ctx: Context, uf: pd.DataFrame, log=print) -> dict:
    rated = rated_films(uf)
    film_model = Hybrid(ctx).fit(rated)
    seen_keys = {item_key(r) for r in uf[uf["watched"] | uf["rating"].notna()].to_dict("records")}
    wl_keys = {item_key(r) for r in uf[uf["watchlist"]].to_dict("records")}
    state = dict(built=time.strftime("%Y-%m-%d %H:%M"), seen=seen_keys, watchlist=wl_keys,
                 film_cv=film_model.cv, film_learners=film_model.describe())

    if ctx.meta:
        keys = [k for k, m in ctx.meta.items() if k.startswith("movie:")]
        T = pd.DataFrame({"key": keys})
        T["ml_id"] = pd.array([ctx.meta[k].get("ml_id") for k in keys], dtype="Int64")
        F = film_model.predict(T)
        F.index = keys
        state["films"] = F
        log(f"scored {len(F):,} films")

        tv_keys = [k for k in ctx.meta if k.startswith("tv:")]
        if tv_keys and ctx.cs is not None:
            shows = _show_training(uf)
            tv_train = pd.concat([rated[["key", "ml_id", "rating", "w"]], shows[["key", "ml_id", "rating", "w"]]]) \
                if len(shows) else rated[["key", "ml_id", "rating", "w"]]
            tv_ctx = Context(cf=None, cs=ctx.cs, meta=ctx.meta)
            tv_model = Hybrid(tv_ctx).fit(tv_train.reset_index(drop=True))
            T = pd.DataFrame({"key": tv_keys, "ml_id": pd.array([None] * len(tv_keys), dtype="Int64")})
            F = tv_model.predict(T)
            F.index = tv_keys
            state["shows"] = F
            state["n_show_ratings"] = int(len(shows))
            state["seen"] |= set(shows["key"]) if len(shows) else set()
            if len(shows):
                s = shows.copy()
                s["name"] = [ctx.meta.get(k, {}).get("title", k) for k in s["key"]]
                s["year"] = [ctx.meta.get(k, {}).get("year") for k in s["key"]]
                s["review"] = ""
                state["tv_rated"] = pd.concat([rated, s], ignore_index=True)
            log(f"scored {len(F):,} shows from {len(shows)} show ratings")
    state["rated"] = rated
    return state


_cache: tuple | None = None  # (mtime, state, context or None), swapped in one assignment


def load_state() -> dict | None:
    global _cache
    if not STATE.exists():
        return None
    mt = STATE.stat().st_mtime
    c = _cache
    if c is None or c[0] != mt:
        with open(STATE, "rb") as f:
            c = (mt, pickle.load(f), None)
        _cache = c
    return c[1]


def context() -> Context:
    global _cache
    load_state()
    c = _cache
    if c is None:
        return build_context()
    if c[2] is None:
        c = (c[0], c[1], build_context())
        _cache = c
    return c[2]
