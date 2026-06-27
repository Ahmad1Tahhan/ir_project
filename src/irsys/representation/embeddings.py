"""Dense embedding representation (sentence-transformers) + FAISS search.

Documents and queries are encoded by the SAME model, so they live in the same
vector space; matching is cosine similarity (via the normalized FAISS index).
Encoding runs on GPU when available (the 4060) — this is the heavy step for
~523K documents, so the embeddings matrix is cached to disk and reused by the
clustering / topic-detection features.
"""
from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np

from .. import config
from ..indexing.vector_store import VectorStore
from .base import BaseRetriever, BuildContext, SearchResult


@lru_cache(maxsize=2)
def get_encoder(model_name: str = config.EMBEDDING_MODEL):
    """Load (and cache) the sentence-transformer, on GPU if available."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model_name, device=device)


def encode_texts(texts: list[str], model_name: str = config.EMBEDDING_MODEL,
                 batch_size: int = 256, show_progress: bool = True) -> np.ndarray:
    encoder = get_encoder(model_name)
    emb = encoder.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
    )
    return emb.astype(np.float32)


class EmbeddingRetriever(BaseRetriever):
    name = "embedding"

    def __init__(self, store: VectorStore, model_name: str, embeddings: np.ndarray | None = None):
        self.store = store
        self.model_name = model_name
        self.embeddings = embeddings  # kept in memory after build; lazy-loaded otherwise
        self._id_to_pos: dict[str, int] | None = None

    @property
    def id_to_pos(self) -> dict[str, int]:
        if self._id_to_pos is None:
            self._id_to_pos = {d: i for i, d in enumerate(self.store.doc_ids)}
        return self._id_to_pos

    def rerank(self, query_clean: str, candidate_doc_ids: list[str]) -> SearchResult:
        """Cosine-rerank a candidate set (used by the serial hybrid). Requires the
        embeddings matrix in memory (load with ``with_embeddings=True``)."""
        if self.embeddings is None:
            raise RuntimeError("rerank needs embeddings in memory; load with with_embeddings=True")
        qv = self._encode_query(query_clean)  # already L2-normalized
        cand = [d for d in candidate_doc_ids if d in self.id_to_pos]
        if not cand:
            return []
        pos = [self.id_to_pos[d] for d in cand]
        sims = self.embeddings[pos] @ qv
        order = np.argsort(-sims)
        return [(cand[o], float(sims[o])) for o in order]

    @classmethod
    def build(cls, ctx: BuildContext, model_name: str = config.EMBEDDING_MODEL,
              batch_size: int = 256) -> "EmbeddingRetriever":
        # Encode the ORIGINAL-ish cleaned text. We use cleaned_texts so the same
        # normalization is applied to docs and queries; transformer models are
        # robust either way, but consistency keeps the pipeline uniform.
        emb = encode_texts(ctx.cleaned_texts, model_name=model_name, batch_size=batch_size)
        store = VectorStore.build(emb, ctx.doc_ids)
        return cls(store, model_name, embeddings=emb)

    def _encode_query(self, query_clean: str) -> np.ndarray:
        return encode_texts([query_clean], model_name=self.model_name, show_progress=False)[0]

    def search(self, query_clean: str, query_tokens: list[str], top_k: int = 10, **kw) -> SearchResult:
        qv = self._encode_query(query_clean)
        return self.store.search(qv, top_k=top_k)

    def query_vector(self, query_clean: str) -> np.ndarray:
        return self._encode_query(query_clean)

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.store.save(d)
        if self.embeddings is not None:
            np.save(d / "embeddings.npy", self.embeddings)
        with open(d / "embedding_meta.pkl", "wb") as f:
            pickle.dump({"model_name": self.model_name}, f)

    @classmethod
    def load(cls, directory: str | Path, with_embeddings: bool = False) -> "EmbeddingRetriever":
        d = Path(directory)
        store = VectorStore.load(d)
        with open(d / "embedding_meta.pkl", "rb") as f:
            meta = pickle.load(f)
        emb = None
        if with_embeddings and (d / "embeddings.npy").exists():
            emb = np.load(d / "embeddings.npy")
        return cls(store, meta["model_name"], embeddings=emb)
