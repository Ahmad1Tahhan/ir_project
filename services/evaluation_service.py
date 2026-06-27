"""Ranking & Evaluation Service.

Serves the pre-computed metric reports (MAP/Recall/P@10/nDCG produced by
scripts/evaluate.py) and can evaluate a single labelled query on demand.

Run standalone:
    uvicorn services.evaluation_service:app --port 8004
"""
from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from irsys import config
from irsys.data import loaders
from irsys.evaluation import evaluate_run
from irsys.evaluation.metrics import average_precision, ndcg_at_k, precision_at_k, recall_at_k
from irsys.pipeline import RetrievalEngine

app = FastAPI(title="IR · Ranking & Evaluation Service")
_engines: dict[str, RetrievalEngine] = {}
_qrels: dict[str, dict] = {}


def get_engine(dataset_key: str) -> RetrievalEngine:
    if dataset_key not in _engines:
        _engines[dataset_key] = RetrievalEngine.load(dataset_key)
    return _engines[dataset_key]


def get_qrels(dataset_key: str) -> dict:
    if dataset_key not in _qrels:
        _qrels[dataset_key] = loaders.load_qrels(dataset_key)
    return _qrels[dataset_key]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/report")
def report(dataset: str = config.DEFAULT_DATASET, tag: str = "baseline"):
    path = config.REPORTS_DIR / f"eval_{dataset}_{tag}.json"
    if not path.exists():
        raise HTTPException(404, f"no report '{tag}' for '{dataset}'. Run scripts/evaluate.py.")
    return json.loads(path.read_text(encoding="utf-8"))


class QueryEvalRequest(BaseModel):
    dataset: str = config.DEFAULT_DATASET
    method: str = "bm25"
    query_id: str
    top_k: int = 100


@app.post("/evaluate_query")
def evaluate_query(req: QueryEvalRequest):
    """Evaluate one labelled query (must exist in qrels)."""
    qrels = get_qrels(req.dataset)
    if req.query_id not in qrels:
        raise HTTPException(404, f"query_id '{req.query_id}' not in qrels")
    queries = loaders.load_queries(req.dataset)
    eng = get_engine(req.dataset)
    ranking = [d for d, _ in eng.search(req.method, queries[req.query_id], top_k=req.top_k)]
    rel = qrels[req.query_id]
    return {
        "query_id": req.query_id,
        "query": queries[req.query_id],
        "method": req.method,
        "AP": average_precision(ranking, rel),
        "P@10": precision_at_k(ranking, rel, 10),
        "nDCG@10": ndcg_at_k(ranking, rel, 10),
        "Recall@100": recall_at_k(ranking, rel, 100),
        "num_relevant": sum(1 for r in rel.values() if r > 0),
    }
