"""Inverted index with on-the-fly BM25 scoring.

This satisfies two requirements at once:
  * Indexing — a real inverted index (term -> postings) for fast retrieval.
  * BM25 with parameters (k1, b) that can be changed *per query at run time*
    (the UI exposes them) because scoring happens at query time, not build time.

Postings are stored column-wise as flat numpy arrays for compactness and speed:
  term_id -> slice into (doc_pos[], tf[]) arrays.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np


class InvertedIndex:
    def __init__(
        self,
        doc_ids: list[str],
        vocab: dict[str, int],
        term_ptr: np.ndarray,     # (V+1,) offsets into postings arrays (CSR-style)
        post_docs: np.ndarray,    # (nnz,) int32 doc positions
        post_tfs: np.ndarray,     # (nnz,) int32 term frequencies
        doc_len: np.ndarray,      # (N,) int32 document lengths (in tokens)
        df: np.ndarray,           # (V,) int32 document frequency per term
    ):
        self.doc_ids = doc_ids
        self.vocab = vocab
        self.term_ptr = term_ptr
        self.post_docs = post_docs
        self.post_tfs = post_tfs
        self.doc_len = doc_len
        self.df = df
        self.N = len(doc_ids)
        self.avgdl = float(doc_len.mean()) if self.N else 0.0
        # BM25 idf (with the +0.5 smoothing, clipped at 0 to avoid negatives)
        self.idf = np.log(1.0 + (self.N - df + 0.5) / (df + 0.5)).astype(np.float32)

    # ----------------------------------------------------------------- build
    @classmethod
    def build(cls, doc_ids: list[str], docs_tokens: list[list[str]]) -> "InvertedIndex":
        vocab: dict[str, int] = {}
        # term_id -> dict(doc_pos -> tf)
        postings: list[dict[int, int]] = []
        doc_len = np.zeros(len(doc_ids), dtype=np.int32)

        for pos, tokens in enumerate(docs_tokens):
            doc_len[pos] = len(tokens)
            local: dict[int, int] = {}
            for tok in tokens:
                tid = vocab.get(tok)
                if tid is None:
                    tid = len(vocab)
                    vocab[tok] = tid
                    postings.append({})
                local[tid] = local.get(tid, 0) + 1
            for tid, tf in local.items():
                postings[tid][pos] = tf

        V = len(vocab)
        df = np.zeros(V, dtype=np.int32)
        nnz = 0
        for tid in range(V):
            df[tid] = len(postings[tid])
            nnz += df[tid]

        term_ptr = np.zeros(V + 1, dtype=np.int64)
        post_docs = np.empty(nnz, dtype=np.int32)
        post_tfs = np.empty(nnz, dtype=np.int32)
        cursor = 0
        for tid in range(V):
            term_ptr[tid] = cursor
            for dpos, tf in postings[tid].items():
                post_docs[cursor] = dpos
                post_tfs[cursor] = tf
                cursor += 1
        term_ptr[V] = cursor

        return cls(doc_ids, vocab, term_ptr, post_docs, post_tfs, doc_len, df)

    # ---------------------------------------------------------------- search
    def bm25_scores(self, query_tokens: list[str], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
        """Full score vector (N,) for a query under given BM25 params."""
        scores = np.zeros(self.N, dtype=np.float32)
        denom_norm = k1 * (1.0 - b + b * self.doc_len / self.avgdl)  # (N,)
        seen: set[int] = set()
        for tok in query_tokens:
            tid = self.vocab.get(tok)
            if tid is None or tid in seen:
                continue
            seen.add(tid)
            s, e = self.term_ptr[tid], self.term_ptr[tid + 1]
            docs = self.post_docs[s:e]
            tfs = self.post_tfs[s:e].astype(np.float32)
            contrib = self.idf[tid] * (tfs * (k1 + 1.0)) / (tfs + denom_norm[docs])
            scores[docs] += contrib
        return scores

    def search(
        self, query_tokens: list[str], top_k: int = 10, k1: float = 1.5, b: float = 0.75
    ) -> list[tuple[str, float]]:
        scores = self.bm25_scores(query_tokens, k1=k1, b=b)
        return self._top_k(scores, top_k)

    def _top_k(self, scores: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        if top_k >= self.N:
            idx = np.argsort(-scores)
        else:
            part = np.argpartition(-scores, top_k)[:top_k]
            idx = part[np.argsort(-scores[part])]
        return [(self.doc_ids[i], float(scores[i])) for i in idx if scores[i] > 0]

    # ----------------------------------------------------------- persistence
    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.savez(
            d / "inverted_index.npz",
            term_ptr=self.term_ptr,
            post_docs=self.post_docs,
            post_tfs=self.post_tfs,
            doc_len=self.doc_len,
            df=self.df,
        )
        with open(d / "inverted_index_meta.pkl", "wb") as f:
            pickle.dump({"doc_ids": self.doc_ids, "vocab": self.vocab}, f)

    @classmethod
    def load(cls, directory: str | Path) -> "InvertedIndex":
        d = Path(directory)
        arr = np.load(d / "inverted_index.npz")
        with open(d / "inverted_index_meta.pkl", "rb") as f:
            meta = pickle.load(f)
        return cls(
            meta["doc_ids"], meta["vocab"], arr["term_ptr"], arr["post_docs"],
            arr["post_tfs"], arr["doc_len"], arr["df"],
        )
