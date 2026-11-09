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

## Offline results (build of 2026-11-09)

Trained on the 667 earliest ratings and tested on the 167 most recent (from 2024-09-20).

| Row | Hybrid | TMDB crowd average | Standard CF |
|---|---|---|---|
| Films with crowd data (69) | RMSE 0.668, rank corr. 0.68, top-quarter 4★+ 100% | 0.663, 0.72, 88% | 0.689, 0.61, 88% |
| New releases (98) | RMSE 0.697, rank corr. 0.58, top-quarter 71% | 0.731, 0.53, 75% | — |
| All (167) | RMSE 0.685, MAE 0.535 | 0.704, 0.558 | 0.782, 0.624 |

It beats standard CF, popularity, TMDB "similar" and your average on every row. Against the plain
TMDB average it wins on error overall and on new releases (better in 91% / 99% of bootstrap
resamples). On films with crowd data it ties on RMSE and trails on rank correlation and AUC. The live
hit rate is the real test.

### Choices and how they were made

Settings were picked on a time-based slice inside the training years (fit on ratings up to
2023-09, validate on 2023-09 → 2024-09), never on the test set:

- Taste signals use your centred rating, not your departure from the crowd (0.623 vs 0.635).
- Recency half-life of 3 years (vs 1.5, 6 and 12).
- One cross-validated blend (ridge won), not an average of ridge and gradient boosting.
- Plain TMDB averages for films (shrinkage tied on validation, so the simpler option). Shows are
  shrunk toward the mean and shifted onto the film scale (TMDB rates TV ~1.2 points higher).

Disclosure: the test set was looked at twice during development (after fixing a MovieLens ID
mapping bug and when checking the shrinkage). Neither changed a setting the validation slice
didn't already support, but the numbers above are not from a perfectly untouched holdout.

Ranking (not prediction) also subtracts uncertainty for low-confidence titles and for titles without
MovieLens data with few TMDB votes, which are often fan-inflated.

## Data

Everything personal stays in `data/` (git-ignored): your export, the TMDB cache, models, and
`reel_taste.sqlite` (feedback, show ratings, block list, recommendation log). Only film and show
lookups go to TMDB. With Claude on, your top and bottom ratings and short review snippets go to
Claude under your own account.
