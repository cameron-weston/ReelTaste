# Reel Taste

Reel Taste is a film and TV recommender built for exactly one person: whoever owns the Letterboxd
account it's pointed at. It reads every film you've logged and rated, learns what *you* like (not
what people in general like), and suggests things you haven't seen yet that it thinks you'll rate
highly.

It runs entirely on your own Mac as a small web app. Nothing about you is uploaded anywhere, except
film lookups to TMDB and, if you switch it on, a short taste summary to Claude for the "why you'll
like it" notes.

## What it's for

Most recommenders are built to keep millions of people watching, so they lean on whatever is
popular. Reel Taste has a narrower goal: predict how many stars you would give a film you haven't
seen, as accurately as possible, and then show you the best ones. It's scored on two things:

1. **Offline:** hide your most recent ratings, train on the rest, and see how close its guesses come
   to what you actually gave. It has to beat the simple alternatives (the TMDB crowd average, a
   standard collaborative-filtering model, "most popular", TMDB's own "similar films", and just
   guessing your average).
2. **Live:** every pick it shows is logged. When you later rate one of those films, it counts as a
   hit if you gave it 4★ or more. That's the number that really matters.

## Getting started

You'll need Python 3.11, Node 18+, and a free TMDB API key.

```bash
python3.11 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.txt
```

Then put your data in place (the `data/` folder is git-ignored, so it never leaves your machine):

- **Your Letterboxd export:** on Letterboxd go to Settings → Data → Export, and unzip it into
  `data/export/`.
- **MovieLens 32M:** download `ml-32m.zip` from grouplens.org and unzip it so the files sit in
  `data/movielens/ml-32m/`. This is the "crowd taste" data: 32 million ratings from 200,000 people.

Create a `.env` file in the project root (also never committed):

| Key | What it's for |
|---|---|
| `TMDB_API_KEY` | Required. Free from themoviedb.org → Settings → API (a v3 key or a v4 read token). |
| `LETTERBOXD_USER` | Your Letterboxd username, so the app can pick up new ratings from your RSS feed. |
| `LLM_BACKEND` | `cli` (default) uses the Claude Code CLI on this Mac and your existing plan; `api` uses an API key. |
| `CLAUDE_CLI` | Optional path to the `claude` binary if it isn't on your `PATH`. |
| `LLM_MODEL` | Which model the CLI uses (default `sonnet`). |
| `ANTHROPIC_API_KEY` | Only needed when `LLM_BACKEND=api`. |

Build everything once (this matches your films to TMDB and MovieLens, downloads catalogs, and trains
the model; the first run takes a while):

```bash
.venv/bin/python -m reel_taste build
```

Then start the app:

```bash
./run.sh
```

and open <http://127.0.0.1:8765>. The first start also builds the web front end.

Claude is optional. To use it with the CLI backend, run `claude` once in a terminal and log in; the
pill in the app's header turns green when it can reach it. Without Claude, everything still works
except the reordering by mood and the written "why".

## Using it

- **Films** and **Shows** are where the recommendations live. Choose *Discover* (anything you haven't
  seen) or *My watchlist* (rank your own watchlist), and optionally type a mood like "a slow-burn
  sci-fi" or "something funny under two hours". The **obscurity slider** decides how much to favour
  lesser-known titles over the obvious picks, and **Filters** narrow by language, years, runtime and
  so on. Page through as many results as you like and choose how many show per page.
- Each pick has buttons to tell the app what happened:
  - **Watched it!** opens a Letterboxd-style log (date, half-star rating, review, rewatch, like).
    Your rating feeds straight into your taste profile. It's only stored here, because Letterboxd
    doesn't let apps post to your diary, so log it there as well if you want it on your profile.
  - **Not interested** hides it from future lists.
  - **+ Watchlist** saves a film for later.
  - **Block** hides that title for good, or everything by its director, or a whole genre.
- **Rate shows you've seen**, on the Shows tab: Letterboxd doesn't cover TV, so rating a few shows here
  gives the TV recommendations something to go on.
- **Your taste** shows what the model has learned about you: where you rate above or below the crowd,
  and the genres, people and eras that pull your ratings up or down.
- **Scoreboard** shows how it's doing: the offline test against every baseline, and the live hit rate
  of the picks you've actually watched.

### Keeping it up to date

The app checks your Letterboxd RSS feed each time it starts (and when you press **Check Letterboxd
RSS** on the Scoreboard). After 10 new ratings it retrains itself.

RSS only shows recent activity and misses edits, so every month or two do a fresh export from
Letterboxd, unzip it over `data/export/`, and press **Full rebuild**.

## How it works

The short version: it combines what millions of other film fans like with what the films themselves
are made of, and then learns, from your ratings alone, how much to trust each of those for *you*.

**1. Crowd taste.** People who share your taste on some films tend to share it on others. Using the
MovieLens ratings, the app learns a compact "taste fingerprint" for every film. It then works out
where *you* sit among those fingerprints from your own ratings (you're placed alongside the crowd,
never mixed into it) and predicts your rating for films you haven't seen.

**2. What a film is made of.** For every film and show it builds a profile from TMDB: genres,
keywords, director, writer, cinematographer, composer, main cast, language, country, decade and plot
summary. Films with similar profiles sit near each other. Each ingredient is weighted evenly, so a
film with a huge cast list doesn't drown out its genre or language. This is also how it handles
brand-new releases and TV shows, which the crowd data doesn't cover.

**3. Clues about each title.** For every candidate it gathers a handful of clues:
- the crowd-taste prediction for you;
- how you rated the films most like it, by crowd taste and by content;
- a taste model built only from content;
- whether TMDB links it to films you loved;
- how the general public rates it.

**4. Weighing the clues.** A small model, trained only on your ratings, learns how much each clue
should count. It's trained so that it's always predicting films it hasn't seen during training,
which keeps it honest. Clues that should point the right way, like "people who like what you like
rated this highly", are never allowed to count against a film. Films too new for the crowd data go
through a separate content-only version.

Every rating counts equally, whenever you logged it. Rewatches count a bit extra, since rewatching
something says you like it. The batch of old films you back-filled on one day in December 2020 counts
a little less, since those ratings were made from memory.

**5. Building the list.** Candidates are ranked by predicted rating, then adjusted:
- the obscurity slider subtracts a bit for popularity (capped, so a film it's very sure you'll love
  can still get through);
- titles it's unsure about need to clear a higher bar;
- a mood, if you typed one, boosts matching titles. Genres you name are required ("sci-fi" means
  sci-fi), and genres you rule out ("no horror") are removed.

A variety pass then makes sure no page is all one director or a run of near-duplicates. With Claude
on, Claude reorders the first dozen to fit your mood and writes a sentence on why each one suits you.

## Does it actually work?

Tested on your 168 most recent ratings (from 17 Sept 2024 on), after training only on the 670
before them. "Error" is how many stars its guesses were off by on average (root mean square; lower
is better). "Order match" is how well its ranking agrees with yours (1 would be perfect).

| | Reel Taste | TMDB crowd average | Standard crowd-taste model |
|---|---|---|---|
| Films with crowd data (71) | error 0.669, order match 0.70; every film in its top quarter you rated 4★+ | 0.699, 0.71, 88% | 0.672, 0.68, 94% |
| New releases (97) | error 0.688, order match 0.56; 71% of its top quarter rated 4★+ | 0.716, 0.54, 75% | n/a |
| Everything (168) | error 0.680 | 0.709 | 0.768 |

It makes smaller errors than the crowd average on every slice, and ranks about as well. It beats
"most popular", TMDB's "similar films" and "just guess your average" everywhere. It also finds 28 of
your own watchlist films in its top 500, against 23 for the crowd average and 15 for the standard
model. The live hit rate on the Scoreboard is the real test, though.

Two honest caveats:
- TMDB averages are today's numbers. When you rated a new release, that consensus didn't exist yet,
  so the crowd-average comparison is flattered a little.
- This test set has been looked at four times during development. No setting was chosen by looking
  at it (see below), but it isn't a perfectly untouched holdout any more.

### How changes get decided

To avoid tuning the model to the test set, every idea is tried on a separate "validation bench"
first (`reel_taste/valbench.py`). It replays two earlier years, Sept 2022 → Sept 2023 and Sept 2023 →
Sept 2024 (252 ratings), each rebuilt to look the way it did at the time: no crowd data that didn't
exist yet, and no rewatches that hadn't happened yet. An idea only ships if it lowers the error
without hurting the ranking, holds up in both years, and wins in at least ~80% of random resamples.

How the current model got here, on that bench (error / order match / how well it separates films you
loved from the rest; the TMDB crowd average scores 0.652 / 0.591 / 0.794):

| Step | Error | Order match | Separation |
|---|---|---|---|
| Starting point | 0.662 | 0.559 | 0.782 |
| Every rating counts equally | 0.633 | 0.602 | 0.812 |
| Crowd clues can't count against a film | 0.631 | 0.609 | 0.819 |
| Every film ingredient weighted evenly | 0.627 | 0.617 | 0.824 |
| Crowd-taste model learns only from real ratings (ALS) | **0.622** | **0.624** | **0.832** |

Ideas that were tried and didn't help: fading out older ratings, recalibrating crowd scores, genre or
decade priors, director and cast priors, finding "taste twins", the EASE item model, boosted trees,
extra interaction clues, using your watchlist and likes as interest signals, and a handful of
alternative ranking methods.

## Commands

```bash
.venv/bin/python -m reel_taste build      # everything: matching, catalogs, content, offline test, scores
```

```bash
.venv/bin/python -m reel_taste retrain    # retrain on your current ratings, reusing catalogs
```

```bash
.venv/bin/python -m reel_taste evaluate   # print the offline scoreboard
```

```bash
.venv/bin/python -m reel_taste cf         # retrain the MovieLens crowd-taste model
```

```bash
.venv/bin/python -m reel_taste.valbench   # score the current model on the validation bench
```

## Where things live

| Path | What's there |
|---|---|
| `reel_taste/` | The Python back end: data loading (`letterboxd.py`, `tmdb.py`, `movielens.py`), the models (`cf.py`, `content.py`, `model.py`), ranking and moods (`recommend.py`, `mood.py`), testing (`evaluate.py`, `valbench.py`) and the web API (`api.py`). |
| `web/` | The React front end. |
| `data/` | Everything personal and everything large: your export, MovieLens, the TMDB cache, trained models, and `reel_taste.sqlite` (your feedback, show ratings, in-app logs, block list and the log of every recommendation). Git-ignored. |
| `.env` | Your keys. Git-ignored. |

## Privacy

Your ratings, reviews and history stay in `data/` on your Mac. TMDB only sees which films and shows
are being looked up. If Claude is on, it receives your highest and lowest ratings and short snippets
of your reviews so it can explain picks, under your own Claude account.
