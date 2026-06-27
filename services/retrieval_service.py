"""Retrieval Service — the heavy service.

Loads a RetrievalEngine per dataset (indexes + embeddings in memory) and serves
ranked results for any representation, with live BM25 params, hybrid mode/fusion
selection, and an optional clustering toggle.

Run standalone:
    uvicorn services.retrieval_service:app --port 8001
"""
from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from irsys import config
from irsys.pipeline import METHODS, RetrievalEngine

app = FastAPI(title="IR · Retrieval Service")
_engines: dict[str, RetrievalEngine] = {}


def get_engine(dataset_key: str) -> RetrievalEngine:
    if dataset_key not in config.DATASETS:
        raise HTTPException(404, f"unknown dataset '{dataset_key}'")
    if dataset_key not in _engines:
        try:
            _engines[dataset_key] = RetrievalEngine.load(dataset_key)
        except FileNotFoundError as e:
            raise HTTPException(409, f"artifacts not built for '{dataset_key}': {e}")
    return _engines[dataset_key]


class SearchRequest(BaseModel):
    dataset: str = config.DEFAULT_DATASET
    method: str = "bm25"
    query: str
    top_k: int = 10
    # BM25 params (live)
    k1: float = 1.5
    b: float = 0.75
    # parallel-hybrid params
    fusion: str = "rrf"           # "rrf" | "weighted"
    candidate_k: int = 100
    # clustering toggle (bonus feature, independently testable)
    use_clustering: bool = False
    n_clusters: int = 1


@app.get("/health")
def health():
    return {"status": "ok", "loaded": list(_engines), "datasets": list(config.DATASETS)}


@app.get("/methods")
def methods():
    return {"methods": METHODS}


@app.post("/search")
def search(req: SearchRequest):
    if req.method not in METHODS:
        raise HTTPException(400, f"unknown method '{req.method}'")
    eng = get_engine(req.dataset)
    t0 = time.time()
    if req.use_clustering:
        ranked = eng.search_clustered(req.query, top_k=req.top_k, n_clusters=req.n_clusters)
        from irsys.data import loaders
        originals = loaders.fetch_original(req.dataset, [d for d, _ in ranked])
        items = [{"doc_id": d, "score": float(s), "text": originals.get(d, "")} for d, s in ranked]
    else:
        params = {"k1": req.k1, "b": req.b, "fusion": req.fusion, "candidate_k": req.candidate_k}
        items = eng.search_with_text(req.method, req.query, top_k=req.top_k, **params)
    return {
        "dataset": req.dataset,
        "method": req.method,
        "use_clustering": req.use_clustering,
        "num_results": len(items),
        "took_ms": round((time.time() - t0) * 1000, 1),
        "results": items,
    }
