"""BM25 retriever — a thin wrapper over the inverted index.

k1 and b are query-time parameters (exposed in the UI) so their effect can be
demonstrated live. Defaults follow the classic BM25 (k1=1.5, b=0.75).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..indexing.inverted_index import InvertedIndex
from .base import BaseRetriever, BuildContext, SearchResult


class BM25Retriever(BaseRetriever):
    name = "bm25"

    def __init__(self, index: InvertedIndex, k1: float = 1.5, b: float = 0.75):
        self.index = index
        self.k1 = k1
        self.b = b

    @classmethod
    def build(cls, ctx: BuildContext, k1: float = 1.5, b: float = 0.75) -> "BM25Retriever":
        index = InvertedIndex.build(ctx.doc_ids, ctx.cleaned_tokens)
        return cls(index, k1=k1, b=b)

    def search(
        self, query_clean: str, query_tokens: list[str], top_k: int = 10,
        k1: float | None = None, b: float | None = None, **kw,
    ) -> SearchResult:
        return self.index.search(
            query_tokens, top_k=top_k,
            k1=self.k1 if k1 is None else k1,
            b=self.b if b is None else b,
        )

    def scores_for(self, query_tokens: list[str], k1: float | None = None, b: float | None = None) -> np.ndarray:
        return self.index.bm25_scores(
            query_tokens, k1=self.k1 if k1 is None else k1, b=self.b if b is None else b
        )

    def save(self, directory: str | Path) -> None:
        self.index.save(directory)

    @classmethod
    def load(cls, directory: str | Path) -> "BM25Retriever":
        return cls(InvertedIndex.load(directory))
