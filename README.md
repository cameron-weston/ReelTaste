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

- **Watched it!** on any pick opens a log: date, rewatch, review, half-star rating, like. It joins your
  ratings like a Letterboxd diary entry (Taste page, live test, training). It is stored here only:
  Letterboxd has no way for apps to post, so log it there too if you want it on your profile. Ratings
  logged as rewatches don't count in the live test, since you'd seen the film before the pick.
- The app reads your Letterboxd RSS feed on start, and from **Scoreboard → Check Letterboxd RSS**.
  After 10 new ratings (films, shows, in-app logs) it retrains on its own.
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

1. **Crowd taste:** MovieLens 32M (200k people, through 2023). Biases plus two sets of item factors:
   ALS on the observed ratings predicts your rating; an SVD gives the vectors for "close to" films.
   You are folded in by ridge regression on your ratings, never mixed into the training data.
2. **Content:** TMDB genres, keywords, director, writer, cinematographer, composer, top cast,
   language/country/decade and plot text, in one space shared by films and shows. Each field is
   normalised on its own, so a long cast or keyword list can't drown out genre and language.
3. **Features per title:** CF prediction, similarity-weighted average of your ratings on the nearest
   films (by crowd taste and by content), a content ridge model of your taste, TMDB recommendation
   links to films you rated, and crowd priors.
4. **Blend:** a ridge trained only on your ratings, out-of-fold, whose weights on the rating signals
   can't go negative. Films MovieLens doesn't know (2024+) go through a content-only blend. Every
   dated rating counts the same, rewatches count extra, and the 4 Dec 2020 backfill a little less.
5. **Ranking:** predicted rating, minus a popularity penalty set by the obscurity dial, plus mood
   match. Then a variety pass (at most two per director, no near-duplicates). With Claude on, it
   reorders the top 50 for your mood and writes the "why".
6. **Testing:** offline, the most recent 20% of ratings by date are held out and every baseline is
   fitted on the same data. Live, every pick shown and every baseline's would-be pick are logged and
   matched to what you later rate.

## Offline results (build of 2026-11-11)

Trained on the 670 earliest ratings and tested on the 168 most recent (from 2024-09-17). RMSE is the
rating error in stars (lower is better); rank corr. is how well the order matches yours.

| Row | Hybrid | TMDB crowd average | Standard CF (MovieLens) |
|---|---|---|---|
| Films with crowd data (71) | RMSE 0.669, rank corr. 0.70, top-quarter 4★+ 100% | 0.699, 0.71, 88% | 0.672, 0.68, 94% |
| New releases (97) | RMSE 0.688, rank corr. 0.56, top-quarter 71% | 0.716, 0.54, 75% | — |
| All (168) | RMSE 0.680, MAE 0.540 | 0.709, 0.565 | 0.768, 0.611 |

It beats the TMDB average on error on every row (better in 98% of bootstrap resamples for new
releases and 96% overall) and ties it on ranking. It beats popularity, TMDB "similar" and your
average everywhere. Standard CF now uses the same ALS factors, so it's a much tougher baseline than
before; the hybrid edges it on films with crowd data (a statistical tie) and is far ahead overall.
On the watchlist check, 28 of your watchlist films land in its top 500 (crowd average 23, CF 15).
The live hit rate is the real test.

### How changes are chosen

`reel_taste/valbench.py` scores every change on two time folds inside the training years
(2022-09 → 2023-09 and 2023-09 → 2024-09; 252 ratings). Each fold is rebuilt as it looked on its start
date: only rewatches logged before it count, and a film only has MovieLens data if it had enough
MovieLens ratings by then. A change ships only if it lowers error without hurting ranking, holds on
both folds, and wins in at least ~80% of paired bootstrap resamples.

Bench results (RMSE / rank corr. / AUC, all 252 ratings; TMDB crowd average 0.652 / 0.591 / 0.794):

| Model | RMSE | Rank corr. | AUC |
|---|---|---|---|
| Before (with the bench fixed) | 0.662 | 0.559 | 0.782 |
| + weights without decay, rescaled to mean 1, blend alpha 100 | 0.633 | 0.602 | 0.812 |
| + rating signals can't get negative weights | 0.631 | 0.609 | 0.819 |
| + each content field normalised on its own | 0.627 | 0.617 | 0.824 |
| + ALS factors for the CF prediction (current) | **0.622** | **0.624** | **0.832** |

Tested and not adopted (tied or lost on the bench): recency decay at any half-life, isotonic or
polynomial crowd calibration, genre/decade crowd priors, a director/cast prior, taste-twin users,
EASE item model, feature-weighted stacking, gradient boosting variants, interaction features,
watched/watchlist "interest" features, a content-to-CF factor map, dropping the content-neighbour
signal, and classifier ranking. The old bench (and the old offline numbers) were flattered: training
weights decayed from today's date, which silently set how strongly the blend was regularised, and
films got MovieLens data in folds dated before it existed.

Disclosure: the offline test set has now been looked at four times during development (after a
MovieLens ID fix, a shrinkage check, a code review, and this round). Settings were chosen on the
bench, and nothing was changed after seeing the test, but it is not a perfectly untouched holdout.

Caveat: TMDB vote averages and counts are today's snapshot. For new releases that consensus didn't
exist yet when you rated them, which flatters the crowd-based baselines (and the hybrid's crowd input).

Ranking (not prediction) also subtracts uncertainty for low-confidence titles and for titles without
MovieLens data with few TMDB votes, which are often fan-inflated.

## Data

Everything personal stays in `data/` (git-ignored): your export, the TMDB cache, models, and
`reel_taste.sqlite` (feedback, show ratings, block list, recommendation log). Only film and show
lookups go to TMDB. With Claude on, your top and bottom ratings and short review snippets go to
Claude under your own account.
