"""Common retriever interface.

Every representation (TF-IDF, BM25, embeddings) and the hybrid combiners
implement this so the retrieval service, evaluation, and UI can treat them
uniformly. A retriever knows how to build itself from a corpus, persist /
reload, and answer queries with ranked (doc_id, score) pairs.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

SearchResult = list[tuple[str, float]]  # [(doc_id, score), ...] best-first


@dataclass
class BuildContext:
    """Inputs shared across representations when building artifacts."""

    dataset_key: str
    doc_ids: list[str]
    cleaned_tokens: list[list[str]]   # preprocessed tokens per doc
    cleaned_texts: list[str]          # space-joined preprocessed text per doc


class BaseRetriever(ABC):
    name: str = "base"

    @abstractmethod
    def search(self, query_clean: str, query_tokens: list[str], top_k: int = 10, **kw) -> SearchResult:
        """Rank documents for one preprocessed query."""

    @abstractmethod
    def save(self, directory: str | Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, directory: str | Path) -> "BaseRetriever": ...
