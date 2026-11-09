"""FastAPI app: JSON API under /api, built front end at /."""
from __future__ import annotations

import threading
import time
import traceback

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import db, insights, llm, pipeline, recommend, tmdb
from .config import RETRAIN_EVERY, ROOT
from .tmdb import IMG

app = FastAPI(title="reel-taste")
_job = {"running": False, "log": [], "error": None, "started": None}


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if o != o else float(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def _run_build(full: bool):
    if _job["running"]:
        return False

    def go():
        _job.update(running=True, log=[], error=None, started=time.strftime("%H:%M:%S"))
        try:
            pipeline.build(full=full, log=lambda s: _job["log"].append(f"{time.strftime('%H:%M:%S')} {s}"))
            pipeline._state_cache.clear()
        except Exception as e:  # noqa: BLE001
            _job["error"] = f"{e}\n{traceback.format_exc()[-1500:]}"
        finally:
            _job["running"] = False

    threading.Thread(target=go, daemon=True).start()
    return True


def _maybe_retrain():
    n = pipeline.refresh_rss()
    if n:
        uf = pipeline.merged_user_films()
        have = int(uf["rating"].notna().sum())
        if have - int(db.get_meta("trained_n_ratings", have)) >= RETRAIN_EVERY:
            _run_build(full=False)
    return n


@app.on_event("startup")
def _startup():
    threading.Thread(target=_maybe_retrain, daemon=True).start()


@app.get("/api/status")
def status():
    st = pipeline.load_state()
    return _clean(dict(
        tmdb_key=tmdb.has_key(), built=bool(st), built_at=db.get_meta("trained_at"),
        has_films=bool(st and "films" in st), has_shows=bool(st and "shows" in st),
        n_show_ratings=(st or {}).get("n_show_ratings", 0),
        rss_last_fetch=db.get_meta("rss_last_fetch"), job={k: _job[k] for k in ("running", "error", "started")},
        job_log=_job["log"][-12:],
    ))


@app.get("/api/llm")
def llm_status(force: bool = False):
    return llm.status(force)


class Rec(BaseModel):
    kind: str = "movie"
    mode: str = "discover"
    mood: str = ""
    filters: dict = {}
    dial: float = 0.5
    n: int = 10
    offset: int = 0
    use_llm: bool = True
    query_id: str | None = None


@app.post("/api/recommend")
def rec(r: Rec):
    return _clean(recommend.recommend(**r.model_dump()))


class Feedback(BaseModel):
    kind: str
    tmdb_id: int
    action: str  # knew | not_interested | watchlist
    query_id: str | None = None


@app.post("/api/feedback")
def feedback(f: Feedback):
    if f.action not in ("knew", "not_interested", "watchlist", "undo"):
        raise HTTPException(400, "bad action")
    db.add_feedback(f.kind, f.tmdb_id, f.action, f.query_id)
    return {"ok": True}


class Block(BaseModel):
    value: str
    label: str = ""


@app.get("/api/blocklist")
def get_blocks():
    return db.blocklist().to_dict("records")


@app.post("/api/blocklist")
def add_block(b: Block):
    db.block(b.value, b.label or b.value)
    return {"ok": True}


@app.delete("/api/blocklist")
def del_block(value: str):
    db.unblock(value)
    return {"ok": True}


@app.get("/api/taste")
def taste():
    return _clean(insights.taste())


@app.get("/api/scoreboard")
def scoreboard():
    return _clean(insights.scoreboard())


@app.post("/api/build")
def build(full: bool = False):
    return {"started": _run_build(full)}


@app.post("/api/refresh")
def refresh():
    return {"new_rss_entries": _maybe_retrain(), "retraining": _job["running"]}


# ---- Shows rating session ------------------------------------------------
@app.get("/api/shows/queue")
def show_queue(n: int = 12, q: str = ""):
    ctx = pipeline.context()
    st = pipeline.load_state() or {}
    done = set(db.show_ratings()["tmdb_id"].astype(int))
    seen = st.get("seen", set())
    items = []
    shows = [m for k, m in ctx.meta.items() if k.startswith("tv:")]
    if q:
        ql = q.lower()
        shows = [m for m in shows if ql in (m.get("title") or "").lower()]
    shows.sort(key=lambda m: -(m.get("vote_count") or 0))
    for m in shows:
        if int(m["id"]) in done or f"tv:{m['id']}" in seen:
            continue
        items.append(dict(tmdb_id=int(m["id"]), title=m.get("title"), year=m.get("year"),
                          poster=f"{IMG}{m['poster']}" if m.get("poster") else None,
                          genres=list(m.get("genres") or []), seasons=m.get("seasons"),
                          networks=list(m.get("networks") or [])))
        if len(items) >= n:
            break
    rated = db.show_ratings()
    return _clean(dict(items=items, n_rated=int(rated["status"].eq("rated").sum()),
                       n_skipped=int((~rated["status"].eq("rated")).sum())))


class ShowRating(BaseModel):
    tmdb_id: int
    rating: float | None = None
    status: str = "rated"  # rated | not_seen


@app.post("/api/shows/rate")
def rate_show(r: ShowRating):
    db.rate_show(r.tmdb_id, r.rating, r.status)
    return {"ok": True}


dist = ROOT / "web" / "dist"
if dist.exists():
    app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        return FileResponse(dist / "index.html")
