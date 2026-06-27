"""VSM with TF-IDF weighting (scikit-learn).

Documents are already preprocessed upstream, so the vectorizer just splits on
whitespace (no extra lowercasing/tokenizing) — this guarantees queries and
documents go through the *identical* pipeline. Vectors are L2-normalized, so a
dot product equals cosine similarity (the matching method for VSM).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .base import BaseRetriever, BuildContext, SearchResult


def _identity(x):
    # Tokens are pre-split lists; tell sklearn not to re-tokenize.
    return x


class TfidfRetriever(BaseRetriever):
    name = "tfidf"

    def __init__(self, vectorizer, doc_matrix: sp.csr_matrix, doc_ids: list[str]):
        self.vectorizer = vectorizer
        self.doc_matrix = doc_matrix          # (N, V) L2-normalized
        self.doc_ids = doc_ids

    @classmethod
    def build(cls, ctx: BuildContext, min_df: int = 2, max_df: float = 0.9) -> "TfidfRetriever":
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(
            analyzer="word",
            tokenizer=_identity,
            preprocessor=_identity,
            token_pattern=None,
            lowercase=False,
            min_df=min_df,
            max_df=max_df,
            norm="l2",
            sublinear_tf=True,
        )
        doc_matrix = vectorizer.fit_transform(ctx.cleaned_tokens)
        return cls(vectorizer, doc_matrix.tocsr(), ctx.doc_ids)

    def search(self, query_clean: str, query_tokens: list[str], top_k: int = 10, **kw) -> SearchResult:
        q = self.vectorizer.transform([query_tokens])           # (1, V), L2-normalized
        scores = (self.doc_matrix @ q.T).toarray().ravel()      # cosine similarity
        return self._top_k(scores, top_k)

    def scores_for(self, query_tokens: list[str]) -> np.ndarray:
        q = self.vectorizer.transform([query_tokens])
        return (self.doc_matrix @ q.T).toarray().ravel()

    def _top_k(self, scores: np.ndarray, top_k: int) -> SearchResult:
        n = scores.shape[0]
        if top_k >= n:
            idx = np.argsort(-scores)
        else:
            part = np.argpartition(-scores, top_k)[:top_k]
            idx = part[np.argsort(-scores[part])]
        return [(self.doc_ids[i], float(scores[i])) for i in idx if scores[i] > 0]

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        sp.save_npz(d / "tfidf_matrix.npz", self.doc_matrix)
        with open(d / "tfidf_model.pkl", "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "doc_ids": self.doc_ids}, f)

    @classmethod
    def load(cls, directory: str | Path) -> "TfidfRetriever":
        d = Path(directory)
        doc_matrix = sp.load_npz(d / "tfidf_matrix.npz").tocsr()
        with open(d / "tfidf_model.pkl", "rb") as f:
            meta = pickle.load(f)
        return cls(meta["vectorizer"], doc_matrix, meta["doc_ids"])
