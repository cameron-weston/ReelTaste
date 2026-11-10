"""Paths and settings. Everything personal lives under data/ and .env, both git-ignored."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
EXPORT_DIR = DATA / "export"
ML_DIR = DATA / "movielens" / "ml-32m"
CACHE = DATA / "cache"
TMDB_CACHE = CACHE / "tmdb"
ARTIFACTS = DATA / "artifacts"
DB_PATH = DATA / "reel_taste.sqlite"

load_dotenv(ROOT / ".env", override=False)

LETTERBOXD_USER = os.getenv("LETTERBOXD_USER", "westonpeace")


def env(name: str, default: str = "") -> str:
    # Re-read .env each call so a key added while the server runs is picked up, without
    # rewriting os.environ from request threads. Falls back to the shell environment.
    return dotenv_values(ROOT / ".env").get(name) or os.getenv(name) or default


for p in (CACHE, TMDB_CACHE, ARTIFACTS):
    p.mkdir(parents=True, exist_ok=True)

# Design constants agreed in planning.
HOLDOUT_FRACTION = 0.20
BACKFILL_DATE = "2020-12-04"  # 326 ratings entered in one sitting; not real watch dates
MIN_RUNTIME = 60
DEFAULT_VOTE_FLOOR = 200
LIVE_BAR = None  # computed from data: share of 4★+ since 2023
RETRAIN_EVERY = 10
