"""Run-time LLM: the Claude Code CLI (default, uses your plan) or the Anthropic API.
Switch with LLM_BACKEND in .env. Everything degrades gracefully to the non-LLM path."""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import httpx

from .config import env

_status_cache: dict = {}


def _cli_path() -> str | None:
    cands = [env("CLAUDE_CLI"), shutil.which("claude"),
             str(Path.home() / ".local/bin/claude"), str(Path.home() / ".claude/local/claude")]
    cands += sorted(glob.glob(str(Path.home() / "Library/Application Support/Claude/claude-code/*/*/claude.app/Contents/MacOS/claude")))[::-1]
    for c in cands:
        if c and os.path.exists(c) and os.access(c, os.X_OK):
            return c
    return None


def _clean_env() -> dict:
    keep = ("HOME", "USER", "LOGNAME", "PATH", "LANG", "SHELL", "TMPDIR")
    e = {k: os.environ[k] for k in keep if k in os.environ}
    e["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
    return e


def _run_cli(prompt: str, timeout: int = 120) -> str:
    exe = _cli_path()
    if not exe:
        raise RuntimeError("Claude Code CLI not found. Install it and run `claude` once to log in.")
    model = env("LLM_MODEL", "sonnet")
    # No tools: the prompt carries third-party text (TMDB overviews), so nothing in it may run anything.
    p = subprocess.run([exe, "-p", prompt, "--output-format", "json", "--model", model, "--tools", ""],
                       capture_output=True, text=True, timeout=timeout, env=_clean_env(), cwd="/tmp")
    try:
        d = json.loads(p.stdout)
    except json.JSONDecodeError:
        raise RuntimeError((p.stderr or p.stdout or "CLI failed")[:300])
    if d.get("is_error"):
        raise RuntimeError(str(d.get("result"))[:300])
    return d.get("result", "")


def _run_api(prompt: str, timeout: int = 120) -> str:
    key = env("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is empty in .env")
    r = httpx.post(env("LLM_API_BASE", "https://api.anthropic.com") + "/v1/messages", timeout=timeout,
                   headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                   json={"model": env("LLM_API_MODEL", "claude-sonnet-5-5"), "max_tokens": 4000,
                         "messages": [{"role": "user", "content": prompt}]})
    r.raise_for_status()
    return "".join(b.get("text", "") for b in r.json()["content"])


def complete(prompt: str, timeout: int = 120) -> str:
    return (_run_api if env("LLM_BACKEND", "cli") == "api" else _run_cli)(prompt, timeout)


def status(force: bool = False) -> dict:
    """Cheap readiness check, cached for 10 minutes."""
    now = time.time()
    if not force and _status_cache.get("ts", 0) > now - 600:
        return _status_cache["v"]
    backend = env("LLM_BACKEND", "cli")
    try:
        if backend == "api":
            ok = bool(env("ANTHROPIC_API_KEY"))
            v = dict(backend="api", ready=ok, detail="API key set" if ok else "ANTHROPIC_API_KEY missing")
        else:
            out = complete("Reply with exactly: ok", timeout=60)
            v = dict(backend="cli", ready="ok" in out.lower(), detail=f"CLI at {_cli_path()}")
    except Exception as e:  # noqa: BLE001
        v = dict(backend=backend, ready=False, detail=str(e)[:200])
    _status_cache.update(ts=now, v=v)
    return v


def parse_json(text: str):
    m = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if m:
        text = m.group(1)
    start = min([i for i in (text.find("["), text.find("{")) if i >= 0], default=0)
    return json.JSONDecoder().raw_decode(text, start)[0]  # ignores any chatter after the JSON
