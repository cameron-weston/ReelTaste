# Reel Taste

A personal film and TV recommender trained on one Letterboxd account. It runs on your Mac and is
judged on two things: predicting your held-out ratings better than standard recommenders, and how
often the picks you actually watch earn 4★ or more.

## Run it

```bash
./run.sh
```

Then open <http://127.0.0.1:8765>. The first run builds the web front end.

## Settings (`.env`, never committed)

| Key | What |
|---|---|
| `TMDB_API_KEY` | Free key from themoviedb.org → Settings → API (v3 key or v4 read token). Required. |
| `LLM_BACKEND` | `cli` (default: Claude Code CLI on this Mac, uses your plan) or `api`. |
| `CLAUDE_CLI` | Optional path to the `claude` binary if it isn't on `PATH`. |
| `LLM_MODEL` | Model alias for the CLI (default `sonnet`). |
| `ANTHROPIC_API_KEY` | Only for `LLM_BACKEND=api`. |

For the CLI backend, run `claude` once in a terminal and log in. The pill in the header turns green
when the app can reach it. Without it, everything works except the Claude reorder and its "why".

## Keeping it current

- The app reads your Letterboxd RSS feed on start, and from **Scoreboard → Check Letterboxd RSS**.
  After 10 new ratings it retrains on its own.
- Every month or two, export from Letterboxd (Settings → Data → Export), unzip it over
  `data/export/`, and press **Full rebuild**. RSS misses edits and old backfills.

## Commands

```bash
.venv/bin/python -m reel_taste build      # full: match, catalogs, content, offline test, scores
.venv/bin/python -m reel_taste retrain    # reuse catalogs, retrain on current ratings
.venv/bin/python -m reel_taste evaluate   # print the offline scoreboard
.venv/bin/python -m reel_taste cf         # retrain the MovieLens CF model
```

## How it works

1. **Crowd taste:** MovieLens 32M (200k people, through 2023). Biases plus a 96-factor SVD; you are
   folded in by ridge regression on your ratings, never mixed into the training data.
2. **Content:** TMDB keywords, genres, director, writer, cinematographer, composer, top cast, language,
   country, decade and plot text, in one space shared by films and shows.
3. **Features per title:** CF prediction, similarity-weighted average of your ratings on the nearest
   films (by crowd taste and by content), a content ridge model of your taste, TMDB recommendation
   links to films you rated, and crowd priors.
4. **Blend:** trained only on your ratings, out-of-fold. Films MovieLens doesn't know (2024+) go
   through a content-only blend. Recent ratings weigh more, rewatches weigh extra, and the
   4 Dec 2020 backfill shares one lower weight.
5. **Ranking:** predicted rating, minus a popularity penalty set by the obscurity dial, plus mood
   match. Then a variety pass (at most two per director, no near-duplicates). With Claude on, it
   reorders the top 50 for your mood and writes the "why".
6. **Testing:** offline, the most recent 20% of ratings by date are held out and every baseline is
   fitted on the same data. Live, every pick shown and every baseline's would-be pick are logged and
   matched to what you later rate.

## Data

Everything personal stays in `data/` (git-ignored): your export, the TMDB cache, models, and
`reel_taste.sqlite` (feedback, show ratings, block list, recommendation log). Only film and show
lookups go to TMDB. With Claude on, your top and bottom ratings and short review snippets go to
Claude under your own account.
