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
    "heist": (["Crime"], ["heist"]),
    "family": (["Family"], ["family"]),
    "violent": (["Action", "Crime"], ["violence", "revenge", "gore"]),
    "bleak": (["Drama", "War"], ["depressing", "tragedy", "despair", "nihilism", "war crimes", "suicide", "genocide", "holocaust"]),
    "dark": ([], ["dark comedy", "depressing", "nihilism", "violence"]),
    "depressing": (["Drama"], ["depressing", "tragedy", "despair"]),
    "heavy": (["Drama", "War"], ["tragedy", "trauma", "grief", "war crimes", "holocaust"]),
    "gory": (["Horror"], ["gore", "splatter", "body horror"]),
    "subtitles": ([], []),
}
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
    m = re.search(r"(?:under|less than|shorter than|max|at most|<)\s*([\d.]+|one|two|three|an|a)\s*(?:and a half\s*)?(hours?|hrs?|h\b|minutes?|mins?|m\b)", t)
    if m:
        n = _num(m.group(1)) + (0.5 if "and a half" in m.group(0) else 0)
        out["filters"]["max_runtime"] = int(n * 60 if m.group(2).startswith("h") else n)
    elif re.search(r"\bshort\b|\bquick\b", t):
        out["filters"]["max_runtime"] = 100
    m = re.search(r"\b(19|20)?(\d)0'?s\b", t)
    if m:
        dec = int((m.group(1) or ("20" if m.group(2) in "012" else "19")) + m.group(2) + "0")
        out["filters"]["year_min"], out["filters"]["year_max"] = dec, dec + 9
    if re.search(r"\b(recent|new|newer|latest)\b", t):
        out["filters"].setdefault("year_min", 2018)
    if re.search(r"\b(old|classic)\b", t) and "year_max" not in out["filters"]:
        out["filters"]["year_max"] = 1979
    if re.search(r"\b(weekend|finish|miniseries|mini-series|limited series|one season|binge)\b", t):
        out["filters"]["max_seasons"] = 1 if re.search(r"\b(miniseries|mini-series|limited|one season)\b", t) else 2
        out["filters"]["finished_only"] = True
    if re.search(r"\b(no subtitles|english)\b", t):
        out["filters"]["languages"] = ["en"]
    # Split into clauses so "funny, not bleak" negates only "bleak".
    for clause in re.split(r"[,;.]| but | and ", t):
        neg = re.search(rf"\b{NEGATORS}\b", clause) is not None
        for word in LEXICON:
            if re.search(rf"\b{re.escape(word)}\b", clause):
                (out["avoid"] if neg else out["want"]).append(word)
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
    return (sum(want) / len(want) if want else 0.0) - (max(avoid) if avoid else 0.0)
