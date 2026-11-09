"""TMDB client with a compact on-disk cache. Only film/show IDs and titles are sent."""
from __future__ import annotations

import asyncio
import gzip
import json
from pathlib import Path

import httpx

from .config import TMDB_CACHE, env

BASE = "https://api.themoviedb.org/3"
IMG = "https://image.tmdb.org/t/p/w342"


class NoKey(RuntimeError):
    pass


def _auth():
    k = env("TMDB_API_KEY").strip()
    if not k:
        raise NoKey("TMDB_API_KEY is empty in .env")
    # v4 read tokens are long JWTs; v3 keys are 32 hex chars.
    if len(k) > 40:
        return {"Authorization": f"Bearer {k}"}, {}
    return {}, {"api_key": k}


def has_key() -> bool:
    return bool(env("TMDB_API_KEY").strip())


def _path(kind: str, id_: int) -> Path:
    d = TMDB_CACHE / kind
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{id_}.json.gz"


def cached(kind: str, id_: int) -> dict | None:
    p = _path(kind, id_)
    if p.exists():
        with gzip.open(p, "rt") as f:
            return json.load(f)
    return None


def _store(kind: str, id_: int, data: dict) -> None:
    with gzip.open(_path(kind, id_), "wt") as f:
        json.dump(data, f)


def _slim(raw: dict, kind: str) -> dict:
    """Keep only what the models and UI use (credits are huge otherwise)."""
    credits = raw.get("credits") or raw.get("aggregate_credits") or {}
    crew = credits.get("crew", [])

    def jobs(*names):
        out = []
        for c in crew:
            j = c.get("job") or ",".join(x.get("job", "") for x in c.get("jobs", []))
            if any(n in j for n in names):
                out.append(c["name"])
        return list(dict.fromkeys(out))[:4]

    kw = raw.get("keywords", {})
    kw = kw.get("keywords") or kw.get("results") or []
    recs = (raw.get("recommendations") or {}).get("results", [])
    sim = (raw.get("similar") or {}).get("results", [])
    base = dict(
        id=raw["id"], kind=kind,
        title=raw.get("title") or raw.get("name"),
        original_title=raw.get("original_title") or raw.get("original_name"),
        date=raw.get("release_date") or raw.get("first_air_date") or "",
        overview=raw.get("overview") or "",
        tagline=raw.get("tagline") or "",
        genres=[g["name"] for g in raw.get("genres", [])],
        keywords=[k["name"] for k in kw][:40],
        cast=[c["name"] for c in credits.get("cast", [])[:10]],
        directors=jobs("Director") if kind == "movie" else [c["name"] for c in raw.get("created_by", [])][:4],
        writers=jobs("Screenplay", "Writer"),
        dop=jobs("Director of Photography"),
        composer=jobs("Original Music Composer", "Music"),
        language=raw.get("original_language"),
        countries=[c.get("iso_3166_1") for c in raw.get("production_countries", [])] or raw.get("origin_country", []),
        runtime=raw.get("runtime") or (raw.get("episode_run_time") or [None])[0],
        vote_average=raw.get("vote_average"), vote_count=raw.get("vote_count"),
        popularity=raw.get("popularity"),
        poster=raw.get("poster_path"),
        imdb_id=(raw.get("external_ids") or {}).get("imdb_id") or raw.get("imdb_id"),
        recommendations=[r["id"] for r in recs][:20],
        similar=[r["id"] for r in sim][:20],
    )
    if kind == "tv":
        base.update(seasons=raw.get("number_of_seasons"), episodes=raw.get("number_of_episodes"),
                    status=raw.get("status"), type=raw.get("type"),
                    networks=[n["name"] for n in raw.get("networks", [])][:3],
                    last_air=raw.get("last_air_date"))
    return base


class Client:
    def __init__(self, concurrency: int = 12):
        self.sem = asyncio.Semaphore(concurrency)
        self.client = httpx.AsyncClient(timeout=20.0)

    async def close(self):
        await self.client.aclose()

    async def get(self, path: str, **params) -> dict:
        headers, auth = _auth()
        for attempt in range(5):
            async with self.sem:
                r = await self.client.get(BASE + path, params={**auth, **params}, headers=headers)
            if r.status_code == 429:
                await asyncio.sleep(1 + attempt * 2)
                continue
            if r.status_code == 404:
                return {}
            r.raise_for_status()
            return r.json()
        r.raise_for_status()
        return {}

    async def details(self, kind: str, id_: int, refresh: bool = False) -> dict | None:
        if not refresh and (c := cached(kind, id_)) is not None:
            return c or None
        extra = "keywords,recommendations,similar,external_ids," + ("credits" if kind == "movie" else "aggregate_credits")
        raw = await self.get(f"/{kind}/{id_}", append_to_response=extra)
        data = _slim(raw, kind) if raw.get("id") else {}
        _store(kind, id_, data)
        return data or None

    async def search(self, kind: str, query: str, year: int | None = None) -> list[dict]:
        params = {"query": query, "include_adult": "false"}
        if year:
            params["year" if kind == "movie" else "first_air_date_year"] = year
        return (await self.get(f"/search/{kind}", **params)).get("results", [])

    async def discover(self, kind: str, pages: int, **params) -> list[dict]:
        out = []
        for p in range(1, pages + 1):
            res = await self.get(f"/discover/{kind}", page=p, **params)
            out += res.get("results", [])
            if p >= res.get("total_pages", 0):
                break
        return out


def run(coro):
    return asyncio.run(coro)
