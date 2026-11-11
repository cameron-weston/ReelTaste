"""Content space shared by films and shows: who made it, what it's about, how it's described."""
from __future__ import annotations

import pickle
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .config import ARTIFACTS

PATH = ARTIFACTS / "content.pkl"


def _slug(s: str) -> str:
    return re.sub(r"\W+", "_", str(s).lower()).strip("_")


def tag_doc(r) -> str:
    toks = []
    toks += [f"g_{_slug(g)}" for g in r["genres"]]
    toks += [f"k_{_slug(k)}" for k in r["keywords"]]
    toks += [f"d_{_slug(d)}" for d in r["directors"]] * 3
    toks += [f"w_{_slug(d)}" for d in r["writers"]] * 2
    toks += [f"p_{_slug(d)}" for d in r["dop"]]
    toks += [f"m_{_slug(d)}" for d in r["composer"]]
    toks += [f"c_{_slug(c)}" for c in list(r["cast"])[:6]]
    if r.get("language"):
        toks.append(f"l_{r['language']}")
    toks += [f"n_{c}" for c in list(r["countries"])[:3]]
    y = r.get("year")
    if y is not None and not pd.isna(y):
        toks.append(f"y_{int(y) // 10 * 10}")
    return " ".join(toks)


def text_doc(r) -> str:
    return f"{r.get('tagline') or ''} {r.get('overview') or ''}"


# Tag prefix -> field block. Language, country and decade share one block.
BLOCKS = {"g": "g", "k": "k", "d": "d", "w": "w", "p": "p", "m": "m", "c": "c", "l": "meta", "n": "meta", "y": "meta"}


def block_normalize(A, names) -> sp.csr_matrix:
    """L2-normalise each field block of a tag matrix on its own, so every field a title has carries
    equal weight: a long keyword or cast list no longer drowns out genre, language and decade.
    (On the validation folds this beat one bag of tags normalised as a whole, on both folds.)"""
    groups = np.unique([BLOCKS[n.split("_", 1)[0]] for n in names], return_inverse=True)[1]
    A = A.tocoo()
    G = groups[A.col]
    sq = sp.csr_matrix((A.data ** 2, (A.row, G)), shape=(A.shape[0], groups.max() + 1)).toarray()
    return sp.csr_matrix((A.data / np.sqrt(sq[A.row, G]), (A.row, A.col)), shape=A.shape)


class ContentSpace:
    def __init__(self):
        self.tags = TfidfVectorizer(token_pattern=r"\S+", min_df=2, sublinear_tf=True, lowercase=False, norm=None)
        self.text = TfidfVectorizer(stop_words="english", min_df=3, max_df=0.3, sublinear_tf=True,
                                    ngram_range=(1, 2), max_features=60000)
        self.svd = TruncatedSVD(128, random_state=0)
        self.index: dict[str, int] = {}
        self.X = None   # sparse, for ridge
        self.Z = None   # dense normalized, for kNN

    def fit(self, frames: dict[str, pd.DataFrame]):
        rows, keys = [], []
        for kind, df in frames.items():
            for r in df.to_dict("records"):
                rows.append(r)
                keys.append(f"{kind}:{r['id']}")
        A = block_normalize(self.tags.fit_transform([tag_doc(r) for r in rows]), self.tags.get_feature_names_out())
        B = self.text.fit_transform([text_doc(r) for r in rows])
        self.X = normalize(sp.hstack([A, B]).tocsr())  # text is one block among the fields
        self.Z = normalize(self.svd.fit_transform(self.X)).astype(np.float32)
        self.index = {k: i for i, k in enumerate(keys)}
        self.vocab = np.array(list(self.tags.get_feature_names_out()) + list(self.text.get_feature_names_out()))
        return self

    def save(self):
        with open(PATH, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load() -> "ContentSpace":
        with open(PATH, "rb") as f:
            return pickle.load(f)

    def rows(self, keys) -> list[int | None]:
        return [self.index.get(k) for k in keys]

    def embed_text(self, s: str) -> np.ndarray:
        """Project free text (e.g. a mood) into the dense space."""
        A = self.tags.transform([""])
        B = self.text.transform([s])
        X = normalize(sp.hstack([A, B]).tocsr())
        return normalize(self.svd.transform(X)).astype(np.float32)[0]
