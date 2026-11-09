"""Match the user's films to TMDB/MovieLens and build the candidate catalogs (films + shows)."""
from __future__ import annotations

import asyncio
import difflib
from datetime import date

import pandas as pd

from . import movielens, tmdb
from .config import ARTIFACTS
from .letterboxd import load_export, norm_title

USER_FILMS = ARTIFACTS / "user_films.parquet"
FILM_CATALOG = ARTIFACTS / "film_catalog.parquet"
TV_CATALOG = ARTIFACTS / "tv_catalog.parquet"

ML_MIN_RATINGS = 30


def _sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, norm_title(a), norm_title(b)).ratio()


def _pick(results: list[dict], name: str, year: int | None, kind: str) -> dict | None:
    best, best_s = None, 0.0
    for r in results[:10]:
        t = r.get("title") or r.get("name") or ""
        ot = r.get("original_title") or r.get("original_name") or ""
        d = r.get("release_date") or r.get("first_air_date") or ""
        ry = int(d[:4]) if d[:4].isdigit() else None
        s = max(_sim(name, t), _sim(name, ot))
        nt, nn = norm_title(t), norm_title(name)
        if len(nn) >= 5 and (nt.startswith(nn + " ") or nn.startswith(nt + " ")):
            s = max(s, 0.8)  # "Glass Onion" vs "Glass Onion: A Knives Out Mystery"
        if year and ry:
            s -= 0.15 * min(abs(ry - year), 3)
        s += min((r.get("vote_count") or 0), 2000) / 20000  # tie-break toward the well-known one
        if s > best_s:
            best, best_s = r, s
    return best if best_s >= 0.72 else None


async def _match_one(c: tmdb.Client, name: str, year: int | None):
    year = None if year is None or pd.isna(year) else int(year)
    mv = _pick(await c.search("movie", name, year), name, year, "movie")
    if mv is None and year:
        mv = _pick(await c.search("movie", name, None), name, year, "movie")
    mv_votes = (mv or {}).get("vote_count") or 0
    tv = None
    if mv is None or mv_votes < 50:
        # Letterboxd lists some miniseries; a near-unknown film with the same name is usually wrong.
        tv = _pick(await c.search("tv", name, year), name, year, "tv") or \
             _pick(await c.search("tv", name, None), name, year, "tv")
    if tv is not None and (mv is None or (tv.get("vote_count") or 0) > 10 * max(mv_votes, 1)):
        return "tv", tv["id"]
    if mv is not None:
        return "movie", mv["id"]
    return None, None


def match_user(log=print) -> pd.DataFrame:
    films = load_export()
    ml = movielens.movies()
    films["ml_id"] = movielens.match_titles(films)
    films["kind"] = None
    films["tmdb_id"] = pd.array([None] * len(films), dtype="Int64")
    if tmdb.has_key():
        prev = pd.read_parquet(USER_FILMS) if USER_FILMS.exists() else None
        known = {} if prev is None or "matched_by" not in prev else {
            r.key: (r.kind if isinstance(r.kind, str) else None, None if pd.isna(r.tmdb_id) else int(r.tmdb_id))
            for r in prev.itertuples() if r.matched_by == "tmdb" and isinstance(r.kind, str)}
        # Re-check film matches that turned out to be near-unknown titles (likely a same-name miniseries).
        for k, (kind, tid) in list(known.items()):
            if kind == "movie" and tid is not None:
                m = tmdb.cached("movie", tid)
                if m is not None and (m.get("vote_count") or 0) < 50:
                    known.pop(k)

        async def go():
            c = tmdb.Client()
            try:
                todo = [f for f in films.itertuples() if f.key not in known]
                log(f"TMDB matching {len(todo)} titles ({len(known)} cached)")
                res = await asyncio.gather(*[_match_one(c, f.name, f.year) for f in todo])
                return {f.key: r for f, r in zip(todo, res)}
            finally:
                await c.close()

        known.update(tmdb.run(go()))
        films["kind"] = films["key"].map(lambda k: known.get(k, (None, None))[0])
        films["tmdb_id"] = pd.array([known.get(k, (None, None))[1] for k in films["key"]], dtype="Int64")
        # Prefer the MovieLens id implied by the TMDB id (more reliable than title matching).
        t2m = movielens.tmdb_to_ml()
        via = films["tmdb_id"].map(lambda t: t2m.get(int(t)) if pd.notna(t) else None)
        films["ml_id"] = pd.array(
            [v if (v is not None and k == "movie") else (m if k != "tv" else None)
             for v, m, k in zip(via, films["ml_id"], films["kind"])], dtype="Int64")
        films["matched_by"] = "tmdb"
    else:
        films["matched_by"] = "title"
        m2t = dict(zip(ml["movieId"], ml["tmdbId"]))
        films["tmdb_id"] = pd.array([m2t.get(int(m)) if pd.notna(m) else None for m in films["ml_id"]], dtype="Int64")
        films["kind"] = films["ml_id"].map(lambda m: "movie" if pd.notna(m) else None)
    films.to_parquet(USER_FILMS)
    return films


def user_films() -> pd.DataFrame:
    return pd.read_parquet(USER_FILMS)


async def _fetch_all(kind: str, ids: list[int], log=print) -> list[dict]:
    c = tmdb.Client(concurrency=16)
    out, done = [], 0
    try:
        for i in range(0, len(ids), 400):
            chunk = ids[i:i + 400]
            res = await asyncio.gather(*[c.details(kind, x) for x in chunk])
            out += [r for r in res if r]
            done += len(chunk)
            log(f"  {kind} details {done}/{len(ids)}")
    finally:
        await c.close()
    return out


def _frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows).drop_duplicates("id")
    df["year"] = pd.to_numeric(df["date"].str[:4], errors="coerce").astype("Int64")
    return df.set_index("id", drop=False)


def build_film_catalog(log=print) -> pd.DataFrame:
    ml = movielens.movies()
    from .cf import CF
    cf = CF.load()
    cnt = dict(zip(cf.item_ids, cf.item_count))
    ml = ml[ml["movieId"].map(lambda m: cnt.get(m, 0)) >= ML_MIN_RATINGS]
    ids = set(ml["tmdbId"].dropna().astype(int))
    uf = user_films()
    ids |= set(uf.loc[uf["kind"].eq("movie"), "tmdb_id"].dropna().astype(int))

    async def recent():
        c = tmdb.Client()
        try:
            out = []
            today = date.today().isoformat()
            for y0, y1 in [("2022-06-01", "2023-12-31"), ("2024-01-01", "2024-12-31"),
                           ("2025-01-01", "2025-12-31"), ("2026-01-01", today)]:
                out += await c.discover("movie", 25, **{
                    "primary_release_date.gte": y0, "primary_release_date.lte": y1,
                    "vote_count.gte": 40, "sort_by": "vote_count.desc",
                    "with_runtime.gte": 60})
            return out
        finally:
            await c.close()

    new = tmdb.run(recent())
    ids |= {r["id"] for r in new}
    log(f"film catalog: {len(ids):,} TMDB ids")
    rows = tmdb.run(_fetch_all("movie", sorted(ids), log))
    df = _frame(rows)
    t2m = movielens.tmdb_to_ml()
    df["ml_id"] = pd.array([t2m.get(int(i)) for i in df["id"]], dtype="Int64")
    df.to_parquet(FILM_CATALOG)
    return df


def build_tv_catalog(log=print) -> pd.DataFrame:
    async def go():
        c = tmdb.Client()
        try:
            res = await c.discover("tv", 60, sort_by="vote_count.desc", **{"vote_count.gte": 150})
            res += await c.discover("tv", 10, sort_by="vote_count.desc", with_genres="16",
                                    with_original_language="ja", **{"vote_count.gte": 60})
            res += await c.discover("tv", 10, sort_by="vote_count.desc", with_genres="99",
                                    **{"vote_count.gte": 40})
            res += await c.discover("tv", 10, sort_by="vote_count.desc",
                                    **{"first_air_date.gte": "2024-01-01", "vote_count.gte": 40})
            return res
        finally:
            await c.close()

    found = tmdb.run(go())
    ids = {r["id"] for r in found}
    uf = user_films()
    ids |= set(uf.loc[uf["kind"].eq("tv"), "tmdb_id"].dropna().astype(int))
    from . import db
    ids |= set(db.show_ratings()["tmdb_id"].astype(int)) if len(db.show_ratings()) else set()
    log(f"tv catalog: {len(ids):,} shows")
    rows = tmdb.run(_fetch_all("tv", sorted(ids), log))
    df = _frame(rows)
    # Talk/news/reality and soaps are not what a "watch a show today" tab is for.
    df = df[~df["genres"].map(lambda g: bool({"Talk", "News", "Soap", "Reality"} & set(g)))]
    df.to_parquet(TV_CATALOG)
    return df


def film_catalog() -> pd.DataFrame:
    return pd.read_parquet(FILM_CATALOG).set_index("id", drop=False)


def tv_catalog() -> pd.DataFrame:
    return pd.read_parquet(TV_CATALOG).set_index("id", drop=False)
