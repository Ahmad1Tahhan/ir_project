"""Hybrid representations — required in BOTH forms:

  * SERIAL (تسلسلي): a pipeline. A fast lexical retriever (e.g. BM25) produces
    a candidate pool, then a semantic retriever (embeddings) re-ranks it.
    Cheap + precise: the expensive model only scores the shortlist.

  * PARALLEL (تفرعي): retrievers run independently and their rankings are merged
    with a fusion method (RRF / weighted sum). Can combine >2 components,
    including multiple embedding models.

Both expose the same ``search`` signature as the base representations, so the
retrieval service and UI switch between them with a single flag.
"""
from __future__ import annotations

from ..representation.base import BaseRetriever, SearchResult
from ..representation.embeddings import EmbeddingRetriever
from .fusion import FUSION_METHODS


class SerialHybridRetriever(BaseRetriever):
    name = "hybrid_serial"

    def __init__(self, first_stage: BaseRetriever, reranker: EmbeddingRetriever, candidate_k: int = 100):
        self.first_stage = first_stage
        self.reranker = reranker
        self.candidate_k = candidate_k

    def search(self, query_clean: str, query_tokens: list[str], top_k: int = 10, **kw) -> SearchResult:
        candidate_k = kw.get("candidate_k", self.candidate_k)
        candidates = self.first_stage.search(query_clean, query_tokens, top_k=candidate_k, **kw)
        if not candidates:
            return []
        reranked = self.reranker.rerank(query_clean, [d for d, _ in candidates])
        return reranked[:top_k]

    def save(self, directory):  # components are persisted independently
        raise NotImplementedError("Hybrid retrievers are composed from saved components.")

    @classmethod
    def load(cls, directory):
        raise NotImplementedError("Hybrid retrievers are composed from saved components.")


class ParallelHybridRetriever(BaseRetriever):
    name = "hybrid_parallel"

    def __init__(
        self,
        components: list[BaseRetriever],
        method: str = "rrf",
        weights: list[float] | None = None,
        candidate_k: int = 100,
    ):
        self.components = components
        self.method = method
        self.weights = weights
        self.candidate_k = candidate_k

    def search(self, query_clean: str, query_tokens: list[str], top_k: int = 10, **kw) -> SearchResult:
        fusion = kw.get("fusion", self.method)
        weights = kw.get("weights", self.weights)
        candidate_k = kw.get("candidate_k", self.candidate_k)
        fuse = FUSION_METHODS[fusion]
        lists = [c.search(query_clean, query_tokens, top_k=candidate_k) for c in self.components]
        fused = fuse(lists, weights=weights)
        return fused[:top_k]

    def save(self, directory):
        raise NotImplementedError("Hybrid retrievers are composed from saved components.")

    @classmethod
    def load(cls, directory):
        raise NotImplementedError("Hybrid retrievers are composed from saved components.")
