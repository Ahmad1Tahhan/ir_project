"""Build + serve orchestration.

``build_artifacts`` runs the full offline pipeline for one dataset:
  load corpus -> preprocess -> build TF-IDF / BM25 / embeddings -> persist.

``RetrievalEngine`` loads those artifacts and serves unified queries across all
representations (including the serial / parallel hybrids), applying the SAME
preprocessing to the query that was used for the documents.
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

from . import config
from .data import loaders
from .preprocessing.text import PreprocessConfig, TextPreprocessor
from .representation.base import BuildContext
from .representation.bm25 import BM25Retriever
from .representation.embeddings import EmbeddingRetriever
from .representation.tfidf import TfidfRetriever
from .retrieval.hybrid import ParallelHybridRetriever, SerialHybridRetriever

METHODS = ["tfidf", "bm25", "embedding", "hybrid_serial", "hybrid_parallel"]


# --------------------------------------------------------------------- build
def build_artifacts(
    dataset_key: str = config.DEFAULT_DATASET,
    limit: int | None = None,
    preprocess_config: PreprocessConfig | None = None,
    embedding_model: str = config.EMBEDDING_MODEL,
    batch_size: int = 256,
) -> dict:
    t0 = time.time()
    pcfg = preprocess_config or PreprocessConfig()
    out_dir = config.artifacts_dir(dataset_key)

    print(f"[build] loading corpus '{dataset_key}' (limit={limit}) ...")
    corpus = loaders.load_corpus(dataset_key, limit=limit)
    print(f"[build] {len(corpus):,} documents")

    print("[build] preprocessing documents ...")
    pre = TextPreprocessor(pcfg)
    cleaned_tokens = [pre.tokens(t) for t in corpus.raw_texts]
    cleaned_texts = [" ".join(toks) for toks in cleaned_tokens]
    ctx = BuildContext(dataset_key, corpus.doc_ids, cleaned_tokens, cleaned_texts)

    print("[build] TF-IDF ...")
    TfidfRetriever.build(ctx).save(out_dir)

    print("[build] BM25 / inverted index ...")
    bm25 = BM25Retriever.build(ctx)
    bm25.save(out_dir)

    print(f"[build] embeddings ({embedding_model}) on GPU ...")
    emb = EmbeddingRetriever.build(ctx, model_name=embedding_model, batch_size=batch_size)
    emb.save(out_dir)

    # Persist build metadata + vocabulary (for query refinement / spelling).
    meta = {
        "dataset_key": dataset_key,
        "num_docs": len(corpus),
        "preprocess_config": pcfg.to_dict(),
        "embedding_model": embedding_model,
        "built_seconds": round(time.time() - t0, 1),
    }
    with open(out_dir / "build_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with open(out_dir / "vocabulary.pkl", "wb") as f:
        pickle.dump(set(bm25.index.vocab.keys()), f)

    print(f"[build] done in {meta['built_seconds']}s -> {out_dir}")
    return meta


# --------------------------------------------------------------------- serve
class RetrievalEngine:
    def __init__(self, dataset_key, preprocessor, tfidf, bm25, embedding, vocabulary, meta):
        self.dataset_key = dataset_key
        self.pre = preprocessor
        self.tfidf = tfidf
        self.bm25 = bm25
        self.embedding = embedding
        self.vocabulary = vocabulary
        self.meta = meta
        self.serial = SerialHybridRetriever(bm25, embedding)
        self.parallel = ParallelHybridRetriever([bm25, embedding])
        self._retrievers = {
            "tfidf": tfidf, "bm25": bm25, "embedding": embedding,
            "hybrid_serial": self.serial, "hybrid_parallel": self.parallel,
        }
        self._clusterer = None  # lazy-loaded (bonus feature, optional)

    @property
    def clusterer(self):
        """Lazy-load the document clusterer if it has been built for this dataset."""
        if self._clusterer is None:
            from .features.clustering import DocumentClusterer
            d = config.artifacts_dir(self.dataset_key)
            if (d / "clusterer.pkl").exists():
                self._clusterer = DocumentClusterer.load(d)
            else:
                raise FileNotFoundError(
                    "Clusterer not built. Run scripts/build_features.py --clustering first."
                )
        return self._clusterer

    def search_clustered(self, query: str, top_k: int = 10, n_clusters: int = 1, **kw):
        """Retrieval restricted to the query's nearest cluster(s) — the
        'with clustering' mode (compare against plain embedding search)."""
        tokens = self.pre.tokens(query)
        clean = " ".join(tokens)
        qv = self.embedding.query_vector(clean)
        candidates = self.clusterer.candidate_doc_ids(qv, n_clusters=n_clusters)
        return self.embedding.rerank(clean, candidates)[:top_k]

    @classmethod
    def load(cls, dataset_key: str = config.DEFAULT_DATASET) -> "RetrievalEngine":
        d = config.artifacts_dir(dataset_key)
        with open(d / "build_meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        pcfg = PreprocessConfig(**meta["preprocess_config"])
        pre = TextPreprocessor(pcfg)
        tfidf = TfidfRetriever.load(d)
        bm25 = BM25Retriever.load(d)
        embedding = EmbeddingRetriever.load(d, with_embeddings=True)  # needed for serial rerank
        with open(d / "vocabulary.pkl", "rb") as f:
            vocab = pickle.load(f)
        return cls(dataset_key, pre, tfidf, bm25, embedding, vocab, meta)

    def search(self, method: str, query: str, top_k: int = 10, **params):
        if method not in self._retrievers:
            raise ValueError(f"unknown method '{method}'. choose from {list(self._retrievers)}")
        tokens = self.pre.tokens(query)
        clean = " ".join(tokens)
        return self._retrievers[method].search(clean, tokens, top_k=top_k, **params)

    def search_with_text(self, method: str, query: str, top_k: int = 10, **params):
        """Search and attach the ORIGINAL (uncleaned) document text + id (for UI)."""
        results = self.search(method, query, top_k=top_k, **params)
        originals = loaders.fetch_original(self.dataset_key, [d for d, _ in results])
        return [
            {"doc_id": d, "score": float(s), "text": originals.get(d, "")}
            for d, s in results
        ]
