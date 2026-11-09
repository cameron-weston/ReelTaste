"""End-to-end build: match → catalogs → content space → offline test → production scores."""
from __future__ import annotations

import pickle
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


def build_context(uf: pd.DataFrame | None = None) -> Context:
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
    """Export + newer RSS entries (ratings, watches, rewatches) merged into one table."""
    uf = catalog.user_films()
    rss = db.rss_entries()
    if len(rss) == 0:
        return uf
    ml = movielens.movies()
    t2m = movielens.tmdb_to_ml()
    by_tmdb = {(k, int(t)): i for i, (k, t) in enumerate(zip(uf["kind"], uf["tmdb_id"])) if pd.notna(t)}
    by_key = {k: i for i, k in enumerate(uf["key"])}
    uf = uf.copy()
    new_rows = []
    for r in rss.sort_values("watched_date").itertuples():
        kind, tid = ("tv", r.tmdb_tv) if pd.notna(r.tmdb_tv) else ("movie", r.tmdb_movie)
        i = by_tmdb.get((kind, int(tid))) if pd.notna(tid) else None
        if i is None:
            i = by_key.get(letterboxd.key(r.name, r.year))
        if i is None:
            new_rows.append(dict(key=letterboxd.key(r.name, r.year), name=r.name,
                                 year=int(r.year) if str(r.year).isdigit() else None,
                                 rating=r.rating, rating_date=r.watched_date, watched=True,
                                 rewatches=int(r.rewatch), liked=bool(r.liked), watchlist=False, review="",
                                 last_watched=r.watched_date, source="rss", backfill=False, kind=kind,
                                 tmdb_id=int(tid) if pd.notna(tid) else None,
                                 ml_id=t2m.get(int(tid)) if (pd.notna(tid) and kind == "movie") else None))
            by_key[new_rows[-1]["key"]] = len(uf) + len(new_rows) - 1
            continue
        if i >= len(uf):
            continue
        uf.at[i, "watched"] = True
        if pd.notna(r.rating) and (pd.isna(uf.at[i, "rating"]) or str(uf.at[i, "rating_date"]) < str(r.watched_date)):
            uf.at[i, "rating"] = r.rating
            uf.at[i, "rating_date"] = r.watched_date
            uf.at[i, "backfill"] = False
        if r.rewatch:
            uf.at[i, "rewatches"] = int(uf.at[i, "rewatches"]) + 1
    if new_rows:
        add = pd.DataFrame(new_rows)
        add["tmdb_id"] = add["tmdb_id"].astype("Int64")
        add["ml_id"] = add["ml_id"].astype("Int64")
        uf = pd.concat([uf, add], ignore_index=True)
    return uf


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
        ctx = build_context(uf)
        log("offline test")
        evaluate(ctx, uf, log)
        log("scoring catalogs")
        state = score_all(ctx, uf, log)
        with open(STATE, "wb") as f:
            pickle.dump(state, f)
        db.set_meta("trained_at", time.strftime("%Y-%m-%d %H:%M"))
        db.set_meta("trained_n_ratings", int(uf["rating"].notna().sum()))
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
        rows.append(dict(key=f"tv:{int(r.tmdb_id)}", rating=r.rating,
                         rating_date=time.strftime("%Y-%m-%d", time.localtime(r.ts)), rewatches=0, backfill=False))
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


_state_cache: dict = {}


def load_state() -> dict | None:
    if not STATE.exists():
        return None
    mt = STATE.stat().st_mtime
    if _state_cache.get("mt") != mt:
        with open(STATE, "rb") as f:
            _state_cache.update(mt=mt, v=pickle.load(f), ctx=None)
    return _state_cache["v"]


def context() -> Context:
    load_state()
    if _state_cache.get("ctx") is None:
        _state_cache["ctx"] = build_context()
    return _state_cache["ctx"]
