"""Read a free-text mood without an LLM: hard limits (runtime, era, language), things to avoid,
and soft mood words mapped onto TMDB genres/keywords. Claude does this better; this is the floor."""
from __future__ import annotations

import re

# mood word -> (genres it favours, keyword fragments it favours)
LEXICON: dict[str, tuple[list[str], list[str]]] = {
    "funny": (["Comedy"], ["comedy", "satire", "parody", "absurd", "farce"]),
    "comedy": (["Comedy"], ["comedy"]),
    "hilarious": (["Comedy"], ["comedy", "absurd", "farce"]),
    "laugh": (["Comedy"], ["comedy"]),
    "warm": (["Comedy", "Family", "Romance"], ["feel-good", "friendship", "heartwarming", "found family", "kindness", "hopeful"]),
    "cozy": (["Comedy", "Family", "Animation"], ["feel-good", "heartwarming", "small town", "friendship"]),
    "comfort": (["Comedy", "Family"], ["feel-good", "heartwarming"]),
    "feel-good": (["Comedy", "Family"], ["feel-good", "heartwarming", "inspirational"]),
    "uplifting": (["Drama", "Family"], ["inspirational", "hopeful", "feel-good"]),
    "romantic": (["Romance"], ["love", "romance", "romantic"]),
    "romance": (["Romance"], ["love", "romance"]),
    "scary": (["Horror"], ["horror", "supernatural", "haunted", "demon"]),
    "horror": (["Horror"], ["horror"]),
    "creepy": (["Horror", "Mystery"], ["eerie", "haunted", "cult", "folk horror"]),
    "tense": (["Thriller"], ["suspense", "tension", "cat and mouse", "survival"]),
    "thriller": (["Thriller"], ["suspense"]),
    "action": (["Action"], ["action", "fight", "martial arts", "heist", "chase"]),
    "fun": (["Comedy", "Adventure", "Action"], ["feel-good", "adventure", "heist", "buddy"]),
    "adventure": (["Adventure"], ["adventure", "quest", "treasure"]),
    "mind-bending": (["Science Fiction", "Mystery"], ["mind-bending", "surreal", "time loop", "nonlinear", "dream"]),
    "strange": ([], ["surreal", "absurd", "avant-garde", "experimental", "dreamlike", "weird"]),
    "weird": ([], ["surreal", "absurd", "avant-garde", "experimental"]),
    "surreal": ([], ["surreal", "dreamlike", "avant-garde"]),
    "slow": (["Drama"], ["slow cinema", "contemplative", "meditative", "minimalism"]),
    "quiet": (["Drama"], ["contemplative", "melancholy", "minimalism"]),
    "sad": (["Drama"], ["grief", "loss", "tragedy", "melancholy"]),
    "cry": (["Drama"], ["grief", "loss", "tearjerker"]),
    "epic": (["History", "War", "Adventure"], ["epic", "historical", "battle"]),
    "sci-fi": (["Science Fiction"], ["space", "future", "dystopia", "artificial intelligence"]),
    "space": (["Science Fiction"], ["space", "astronaut", "outer space"]),
    "animated": (["Animation"], ["anime", "animation"]),
    "anime": (["Animation"], ["anime"]),
    "documentary": (["Documentary"], ["documentary"]),
    "true crime": (["Documentary", "Crime"], ["true crime", "murder"]),
    "musical": (["Music"], ["musical"]),
    "western": (["Western"], ["western"]),
    "mystery": (["Mystery"], ["whodunit", "investigation", "detective"]),
    "noir": (["Crime"], ["film noir", "neo-noir", "detective"]),
    "crime": (["Crime"], ["heist", "gangster", "organized crime", "police", "robbery"]),
    "heist": (["Crime"], ["heist"]),
    "family": (["Family"], ["family"]),
    "violent": (["Action", "Crime"], ["violence", "revenge", "gore"]),
    "bleak": (["Drama", "War"], ["depressing", "tragedy", "despair", "nihilism", "war crimes", "suicide", "genocide", "holocaust"]),
    "dark": ([], ["dark comedy", "depressing", "nihilism", "violence"]),
    "depressing": (["Drama"], ["depressing", "tragedy", "despair"]),
    "heavy": (["Drama", "War"], ["tragedy", "trauma", "grief", "war crimes", "holocaust"]),
    "gory": (["Horror"], ["gore", "splatter", "body horror"]),
    "subtitles": ([], []),
    "science fiction": (["Science Fiction"], ["space", "future", "dystopia", "artificial intelligence"]),
    "scifi": (["Science Fiction"], ["space", "future", "dystopia", "artificial intelligence"]),
    "sci fi": (["Science Fiction"], ["space", "future", "dystopia", "artificial intelligence"]),
    "fantasy": (["Fantasy"], ["magic", "fantasy world", "dragon", "sword and sorcery"]),
    "war": (["War"], ["war", "battle", "soldier"]),
    "moody": ([], ["atmospheric", "melancholy", "loneliness", "neo-noir", "existentialism", "isolation"]),
    "atmospheric": ([], ["atmospheric", "eerie", "melancholy", "dreamlike"]),
    "cerebral": ([], ["philosophy", "existentialism", "consciousness", "identity", "memory", "mind-bending",
                      "artificial intelligence", "time paradox"]),
    "thoughtful": ([], ["philosophy", "existentialism", "consciousness", "identity", "memory"]),
    "makes you think": ([], ["philosophy", "existentialism", "consciousness", "identity", "morality"]),
    "thought-provoking": ([], ["philosophy", "existentialism", "consciousness", "identity", "morality"]),
    "gore": (["Horror"], ["gore", "splatter", "body horror"]),
    "kids": (["Family"], ["children", "kids"]),
    "slow burn": (["Drama", "Thriller"], ["slow burn", "slow cinema", "contemplative", "atmospheric", "meditative"]),
}

# Words that name a genre. A wanted one is a requirement, an avoided one an exclusion; tone words
# ("dark", "slow") stay soft. Sets include TMDB's TV genre names.
_SCIFI = {"Science Fiction", "Sci-Fi & Fantasy"}
GENRE_WORDS: dict[str, set[str]] = {
    "sci-fi": _SCIFI, "science fiction": _SCIFI, "scifi": _SCIFI, "sci fi": _SCIFI, "space": _SCIFI,
    "fantasy": {"Fantasy", "Sci-Fi & Fantasy"},
    "horror": {"Horror"},
    "comedy": {"Comedy"},
    "romance": {"Romance"},
    "western": {"Western"},
    "anime": {"Animation"}, "animated": {"Animation"},
    "documentary": {"Documentary"}, "true crime": {"Documentary"},
    "musical": {"Music"},
    "war": {"War", "War & Politics"},
    "mystery": {"Mystery"},
    "crime": {"Crime"}, "noir": {"Crime", "Mystery", "Thriller"}, "heist": {"Crime"},
    "thriller": {"Thriller", "Crime", "Mystery"},
    "action": {"Action", "Action & Adventure"},
    "adventure": {"Adventure", "Action & Adventure"},
    "family": {"Family", "Kids"}, "kids": {"Family", "Kids"},
}


# Tone words that a genre contradicts: "dark and moody" rules against a comedy even when it's sci-fi.
_LIGHT = {"Comedy", "Family", "Kids"}
OPPOSED: dict[str, set[str]] = {
    "dark": _LIGHT, "moody": _LIGHT, "bleak": _LIGHT, "depressing": _LIGHT, "heavy": _LIGHT,
    "cerebral": {"Family", "Kids"}, "makes you think": {"Family", "Kids"}, "thoughtful": {"Family", "Kids"},
    "thought-provoking": {"Family", "Kids"}, "slow burn": {"Action", "Comedy"}, "slow": {"Action"},
    "tense": {"Family", "Kids"}, "scary": _LIGHT, "creepy": _LIGHT,
    "funny": {"Horror", "War"}, "warm": {"Horror", "War"}, "cozy": {"Horror", "War", "Thriller"},
    "comfort": {"Horror", "War", "Thriller"}, "feel-good": {"Horror", "War"}, "uplifting": {"Horror", "War"},
}


def genre_rules(parsed: dict) -> tuple[list[set[str]], set[str]]:
    """(required: one genre set per wanted genre word, excluded: genres of avoided genre words)."""
    req, seen = [], set()
    for w in parsed["want"]:
        g = GENRE_WORDS.get(w)
        if g and frozenset(g) not in seen:
            seen.add(frozenset(g))
            req.append(g)
    excl: set[str] = set()
    for w in parsed.get("hard_avoid", parsed["avoid"]):
        excl |= GENRE_WORDS.get(w, set())
    return req, excl
NEGATORS = r"(?:not|no|nothing|without|avoid|skip|less|isn'?t|aren'?t|never|but not)"
NUMBERS = {"one": 1, "two": 2, "three": 3, "an": 1, "a": 1}


def _num(s: str) -> float:
    s = s.strip().lower()
    if s in NUMBERS:
        return NUMBERS[s]
    try:
        return float(s.replace("½", ".5"))
    except ValueError:
        return 0


def parse(text: str) -> dict:
    t = (text or "").lower()
    out: dict = {"filters": {}, "want": [], "avoid": []}
    m = re.search(r"(?:under|less than|shorter than|max|at most|<)\s*([\d.]+|one|two|three|an|a)\s*(?:and a half\s*)?(hours?|hrs?|h\b|minutes?|mins?|m\b)(?:\s*and a half)?", t)
    if m:
        n = _num(m.group(1)) + (0.5 if "and a half" in m.group(0) else 0)
        out["filters"]["max_runtime"] = int(n * 60 if m.group(2).startswith("h") else n)
    elif re.search(r"\bshort\b|\bquick\b", t):
        out["filters"]["max_runtime"] = 100
    m = re.search(r"\b(19|20)?(\d)0'?s\b", t)
    if m:
        dec = int((m.group(1) or ("20" if m.group(2) in "012" else "19")) + m.group(2) + "0")
        out["filters"]["year_min"], out["filters"]["year_max"] = dec, dec + 9
    # Era words, per clause so "nothing old" or "no new stuff" don't set the opposite filter.
    # "new" as in recent, not New York / New Wave / new to me; "classic" but not "modern classic".
    for clause in re.split(r"[,;.]| but ", t):
        if re.search(rf"\b{NEGATORS}\b", clause):
            continue
        if re.search(r"\b(recent|newer|latest)\b|\bnew\b(?!\s+(?:york|zealand|orleans|jersey|mexico|wave|hollywood|to me))", clause):
            out["filters"].setdefault("year_min", 2018)
        if re.search(r"\bold\b|(?<!modern )\bclassic\b", clause) and "year_max" not in out["filters"]:
            out["filters"]["year_max"] = 1979
    if re.search(r"\b(weekend|finish|miniseries|mini-series|limited series|one season|binge)\b", t):
        out["filters"]["max_seasons"] = 1 if re.search(r"\b(miniseries|mini-series|limited|one season)\b", t) else 2
        out["filters"]["finished_only"] = True
    for clause in re.split(r"[,;.]| but ", t):  # "not in english" means the opposite
        if (re.search(r"\b(no subtitles|english)\b", clause)
                and not re.search(r"\b(?:not|no|non)[\s-]+(?:in\s+)?english\b", clause)):
            out["filters"]["languages"] = ["en"]
    # Split into clauses so "funny, not bleak" negates only "bleak".
    # A negator only covers the words after it: "sci-fi without romance" wants sci-fi, avoids romance.
    # "less" is a soft ask (down-weight), every other negator also hard-excludes a named genre.
    out["hard_avoid"] = []
    for clause in re.split(r"[,;.]| but | and ", t):
        m = re.search(rf"\b{NEGATORS}\b", clause)
        for word in LEXICON:
            w = re.search(rf"\b{re.escape(word)}\b", clause)
            if not w:
                continue
            if m and w.start() > m.start():
                out["avoid"].append(word)
                if m.group(0) != "less":
                    out["hard_avoid"].append(word)
            else:
                out["want"].append(word)
    if "true crime" in out["want"] and "crime" in out["want"]:
        out["want"].remove("crime")
    return out


def score(meta: dict, parsed: dict) -> float:
    """Soft mood fit in roughly [-1, 1]."""
    g = set(meta.get("genres") or [])
    kw = " | ".join(meta.get("keywords") or []).lower()

    def hit(word):
        gs, ks = LEXICON[word]
        s = 0.0
        if gs and g & set(gs):
            s += 0.5
        s += 0.5 * min(1, sum(k in kw for k in ks))
        return s
    want = [hit(w) for w in parsed["want"]]
    avoid = [hit(w) for w in parsed["avoid"]]
    clash = 0.6 if any(g & OPPOSED.get(w, set()) for w in parsed["want"]) else 0.0
    return (sum(want) / len(want) if want else 0.0) - (max(avoid) if avoid else 0.0) - clash
