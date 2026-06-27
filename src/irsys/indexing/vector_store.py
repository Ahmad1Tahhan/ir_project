"""FAISS vector store (the "use vector stores" bonus feature).

Stores L2-normalized document embeddings in a FAISS index so cosine-similarity
search is exact and fast. IndexFlatIP on normalized vectors == cosine. For
~523K x 384 float32 (~0.8 GB) flat search is a few ms per query; swap to
IVF/HNSW here if a larger corpus needs sub-linear search.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    np.divide(x, norms, out=x, where=norms > 0)
    return x


class VectorStore:
    def __init__(self, index, doc_ids: list[str], dim: int):
        self.index = index
        self.doc_ids = doc_ids
        self.dim = dim

    @classmethod
    def build(cls, embeddings: np.ndarray, doc_ids: list[str]) -> "VectorStore":
        import faiss

        emb = _l2_normalize(embeddings)
        dim = emb.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(emb)
        return cls(index, doc_ids, dim)

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        q = _l2_normalize(query_vec.reshape(1, -1))
        sims, idx = self.index.search(q, min(top_k, len(self.doc_ids)))
        out = []
        for score, i in zip(sims[0], idx[0]):
            if i < 0:
                continue
            out.append((self.doc_ids[i], float(score)))
        return out

    def save(self, directory: str | Path) -> None:
        import faiss

        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(d / "vectors.faiss"))
        with open(d / "vectors_meta.pkl", "wb") as f:
            pickle.dump({"doc_ids": self.doc_ids, "dim": self.dim}, f)

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        import faiss

        d = Path(directory)
        index = faiss.read_index(str(d / "vectors.faiss"))
        with open(d / "vectors_meta.pkl", "rb") as f:
            meta = pickle.load(f)
        return cls(index, meta["doc_ids"], meta["dim"])
