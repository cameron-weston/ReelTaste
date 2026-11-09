"""MovieLens 32M: titles, TMDB links, and a cached compact ratings array."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import ARTIFACTS, ML_DIR
from .letterboxd import norm_title

_YEAR = re.compile(r"^(.*)\s+\((\d{4})\)\s*$")
_AKA = re.compile(r"\s*\((?:a\.k\.a\.\s*)?([^()]*)\)")


def movies() -> pd.DataFrame:
    p = ARTIFACTS / "ml_movies.parquet"
    if p.exists():
        return pd.read_parquet(p)
    m = pd.read_csv(ML_DIR / "movies.csv")
    links = pd.read_csv(ML_DIR / "links.csv")
    m = m.merge(links, on="movieId", how="left")
    titles, years, alts = [], [], []
    for t in m["title"]:
        mm = _YEAR.match(t.strip())
        base, yr = (mm.group(1), int(mm.group(2))) if mm else (t, None)
        alt = [a for a in _AKA.findall(base)]
        base = _AKA.sub("", base).strip()
        titles.append(base)
        years.append(yr)
        alts.append("|".join(norm_title(a) for a in alt))
    m["base_title"] = titles
    m["year"] = years
    m["norm"] = [norm_title(t) for t in titles]
    m["alt_norm"] = alts
    m["tmdbId"] = m["tmdbId"].astype("Int64")
    m.to_parquet(p)
    return m


def tmdb_to_ml() -> dict[int, int]:
    m = movies().dropna(subset=["tmdbId"])
    return dict(zip(m["tmdbId"].astype(int), m["movieId"].astype(int)))


def ratings() -> pd.DataFrame:
    """userId, movieId (int32), rating (float32). Cached as parquet after first load."""
    p = ARTIFACTS / "ml_ratings.parquet"
    if p.exists():
        return pd.read_parquet(p)
    r = pd.read_csv(ML_DIR / "ratings.csv", usecols=["userId", "movieId", "rating"],
                    dtype={"userId": np.int32, "movieId": np.int32, "rating": np.float32})
    r.to_parquet(p)
    return r


def match_titles(films: pd.DataFrame) -> pd.Series:
    """Fallback matching of Letterboxd films to MovieLens by normalised title + year (±1)."""
    m = movies()
    idx: dict[tuple[str, int], int] = {}
    for row in m.itertuples():
        if row.year is None or pd.isna(row.year):
            continue
        for n in [row.norm, *[a for a in str(row.alt_norm).split("|") if a]]:
            idx.setdefault((n, int(row.year)), row.movieId)
    out = []
    for f in films.itertuples():
        n = norm_title(f.name)
        hit = None
        if f.year and not pd.isna(f.year):
            for dy in (0, -1, 1):
                hit = idx.get((n, int(f.year) + dy))
                if hit:
                    break
        out.append(hit)
    return pd.Series(out, index=films.index, dtype="Int64")
