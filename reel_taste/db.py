"""Local SQLite: show ratings, feedback, recommendation log (for the live test), block list."""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager

import pandas as pd

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS show_ratings(tmdb_id INTEGER PRIMARY KEY, rating REAL, status TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS queries(query_id TEXT PRIMARY KEY, ts REAL, kind TEXT, mode TEXT, mood TEXT,
    filters TEXT, dial REAL, llm INTEGER);
CREATE TABLE IF NOT EXISTS rec_log(query_id TEXT, ts REAL, kind TEXT, source TEXT, rank INTEGER,
    tmdb_id INTEGER, predicted REAL, shown INTEGER);
CREATE INDEX IF NOT EXISTS rec_log_id ON rec_log(kind, tmdb_id);
CREATE TABLE IF NOT EXISTS feedback(ts REAL, kind TEXT, tmdb_id INTEGER, action TEXT, query_id TEXT);
CREATE TABLE IF NOT EXISTS blocklist(value TEXT PRIMARY KEY, label TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS rss_entries(guid TEXT PRIMARY KEY, name TEXT, year TEXT, rating REAL,
    watched_date TEXT, rewatch INTEGER, liked INTEGER, tmdb_movie INTEGER, tmdb_tv INTEGER, ts REAL);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
"""


@contextmanager
def conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    try:
        c.executescript(SCHEMA)
        yield c
        c.commit()
    finally:
        c.close()


def q(sql: str, *args) -> pd.DataFrame:
    with conn() as c:
        return pd.read_sql_query(sql, c, params=args)


def show_ratings() -> pd.DataFrame:
    return q("SELECT * FROM show_ratings")


def rate_show(tmdb_id: int, rating: float | None, status: str):
    with conn() as c:
        c.execute("INSERT OR REPLACE INTO show_ratings VALUES(?,?,?,?)", (tmdb_id, rating, status, time.time()))


def new_query(kind, mode, mood, filters, dial, llm) -> str:
    qid = uuid.uuid4().hex[:12]
    with conn() as c:
        c.execute("INSERT INTO queries VALUES(?,?,?,?,?,?,?,?)",
                  (qid, time.time(), kind, mode, mood, json.dumps(filters), dial, int(llm)))
    return qid


def log_recs(qid: str, kind: str, source: str, items: list[tuple[int, float]], shown: bool):
    ts = time.time()
    with conn() as c:
        c.executemany("INSERT INTO rec_log VALUES(?,?,?,?,?,?,?,?)",
                      [(qid, ts, kind, source, i + 1, int(t), float(p) if p == p else None, int(shown))
                       for i, (t, p) in enumerate(items)])


def add_feedback(kind, tmdb_id, action, qid=None):
    with conn() as c:
        c.execute("INSERT INTO feedback VALUES(?,?,?,?,?)", (time.time(), kind, tmdb_id, action, qid))


def feedback() -> pd.DataFrame:
    return q("SELECT * FROM feedback")


def block(value: str, label: str):
    with conn() as c:
        c.execute("INSERT OR REPLACE INTO blocklist VALUES(?,?,?)", (value, label, time.time()))


def unblock(value: str):
    with conn() as c:
        c.execute("DELETE FROM blocklist WHERE value=?", (value,))


def blocklist() -> pd.DataFrame:
    return q("SELECT * FROM blocklist ORDER BY ts DESC")


def get_meta(k: str, default=None):
    with conn() as c:
        r = c.execute("SELECT value FROM meta WHERE key=?", (k,)).fetchone()
    return json.loads(r[0]) if r else default


def set_meta(k: str, v):
    with conn() as c:
        c.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (k, json.dumps(v)))


def save_rss(df: pd.DataFrame) -> int:
    """Insert RSS entries; returns how many were new."""
    new = 0
    with conn() as c:
        for r in df.itertuples():
            guid = f"{r.name}|{r.year}|{r.watched_date}|{r.rating}"
            cur = c.execute("INSERT OR IGNORE INTO rss_entries VALUES(?,?,?,?,?,?,?,?,?,?)",
                            (guid, r.name, r.year, None if r.rating != r.rating else r.rating,
                             r.watched_date, int(r.rewatch), int(r.liked),
                             int(r.tmdb_movie) if pd.notna(r.tmdb_movie) and r.tmdb_movie else None,
                             int(r.tmdb_tv) if pd.notna(r.tmdb_tv) and r.tmdb_tv else None, time.time()))
            new += cur.rowcount
    return new


def rss_entries() -> pd.DataFrame:
    return q("SELECT * FROM rss_entries")
