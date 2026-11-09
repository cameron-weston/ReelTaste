"""Parse the Letterboxd export (and RSS) into one row per film the user has touched."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from .config import BACKFILL_DATE, EXPORT_DIR, LETTERBOXD_USER


def _read(name: str) -> pd.DataFrame:
    p = EXPORT_DIR / f"{name}.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, dtype=str, keep_default_na=False)


def key(name: str, year) -> str:
    return f"{str(name).strip().lower()}|{str(year).strip()}"


def load_export() -> pd.DataFrame:
    """One row per film: rating, watched, rewatch count, liked, watchlisted, review text."""
    watched = _read("watched")
    ratings = _read("ratings")
    diary = _read("diary")
    reviews = _read("reviews")
    watchlist = _read("watchlist")
    likes = _read("likes/films")

    films: dict[str, dict] = {}

    def row(name, year, uri=None):
        k = key(name, year)
        if k not in films:
            films[k] = dict(key=k, name=name, year=int(year) if str(year).isdigit() else None,
                            uri=uri, rating=np.nan, rating_date=None, watched=False,
                            rewatches=0, liked=False, watchlist=False, review="",
                            last_watched=None, source="export")
        elif uri and not films[k]["uri"]:
            films[k]["uri"] = uri
        return films[k]

    for r in watched.itertuples():
        f = row(r.Name, r.Year, r._4)
        f["watched"] = True
        f["last_watched"] = max(filter(None, [f["last_watched"], r.Date]))
    for r in ratings.itertuples():
        f = row(r.Name, r.Year, r._4)
        f["rating"] = float(r.Rating)
        f["rating_date"] = r.Date
        f["watched"] = True
    for r in diary.itertuples():
        f = row(r.Name, r.Year)
        f["watched"] = True
        if r.Rewatch:
            f["rewatches"] += 1
        wd = r._8 or r.Date
        f["last_watched"] = max(filter(None, [f["last_watched"], wd]))
    for r in reviews.itertuples():
        f = row(r.Name, r.Year)
        if r.Review:
            f["review"] = (f["review"] + "\n" + r.Review).strip()
    for r in watchlist.itertuples():
        f = row(r.Name, r.Year, r._4)
        f["watchlist"] = True
    for r in likes.itertuples():
        row(r.Name, r.Year, r._4)["liked"] = True

    df = pd.DataFrame(films.values())
    df["backfill"] = df["rating_date"].eq(BACKFILL_DATE)
    return df


def evidence_weight(df: pd.DataFrame, today: str | None = None) -> pd.Series:
    """Training weight per rated film: recent ratings count slightly more; the backfill
    day shares one lower weight; rewatches are a strong boost."""
    today = pd.Timestamp(today or pd.Timestamp.today().date())
    d = pd.to_datetime(df["rating_date"], errors="coerce")
    years_ago = ((today - d).dt.days / 365.25).fillna(6.0)
    w = 0.5 ** (years_ago / 6.0)  # half-life of six years: gentle drift tracking
    w = w.where(~df["backfill"], 0.5 ** (6.0 / 6.0) * 0.85)
    w = w * (1 + 0.5 * np.minimum(df["rewatches"], 2))
    return w


def fetch_rss(user: str = LETTERBOXD_USER, timeout: float = 15.0) -> pd.DataFrame:
    """Recent diary entries from the public RSS feed (includes TMDB ids)."""
    url = f"https://letterboxd.com/{user}/rss/"
    r = httpx.get(url, timeout=timeout, follow_redirects=True,
                  headers={"User-Agent": "reel-taste/0.1 (personal recommender)"})
    r.raise_for_status()
    root = ET.fromstring(r.content)
    ns = {"letterboxd": "https://letterboxd.com", "tmdb": "https://themoviedb.org"}
    out = []
    for it in root.iter("item"):
        title = it.findtext("letterboxd:filmTitle", namespaces=ns)
        if not title:
            continue  # lists etc.
        rating = it.findtext("letterboxd:memberRating", namespaces=ns)
        out.append(dict(
            name=title,
            year=it.findtext("letterboxd:filmYear", namespaces=ns),
            rating=float(rating) if rating else np.nan,
            watched_date=it.findtext("letterboxd:watchedDate", namespaces=ns),
            rewatch=(it.findtext("letterboxd:rewatch", namespaces=ns) or "").lower() == "yes",
            tmdb_movie=it.findtext("tmdb:movieId", namespaces=ns),
            tmdb_tv=it.findtext("tmdb:tvId", namespaces=ns),
            liked=(it.findtext("letterboxd:memberLike", namespaces=ns) or "").lower() == "yes",
        ))
    return pd.DataFrame(out)


_ARTICLE = re.compile(r"^(.*), (the|a|an|les|la|le|l'|el|il|der|die|das)$", re.I)


def norm_title(t: str) -> str:
    t = t.strip()
    m = _ARTICLE.match(t)
    if m:
        t = f"{m.group(2)} {m.group(1)}"
    t = t.lower().replace("&", "and")
    t = re.sub(r"[^\w\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()
