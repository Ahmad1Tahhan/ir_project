"""Query Refinement Service — spelling correction, synonym expansion,
suggestions. Independently testable.

Run standalone:
    uvicorn services.refinement_service:app --port 8003
"""
from __future__ import annotations

import pickle

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from irsys import config
from irsys.data import loaders
from irsys.preprocessing.text import PreprocessConfig, TextPreprocessor
from irsys.refinement import QueryRefiner

app = FastAPI(title="IR · Query Refinement Service")
_refiners: dict[str, QueryRefiner] = {}
_pre = TextPreprocessor(PreprocessConfig())


def get_refiner(dataset_key: str) -> QueryRefiner:
    if dataset_key not in config.DATASETS:
        raise HTTPException(404, f"unknown dataset '{dataset_key}'")
    if dataset_key not in _refiners:
        d = config.artifacts_dir(dataset_key)
        vocab_path = d / "vocabulary.pkl"
        if not vocab_path.exists():
            raise HTTPException(409, f"artifacts not built for '{dataset_key}'")
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        # Use the dataset's own queries as the suggestion pool.
        try:
            query_log = list(loaders.load_queries(dataset_key).values())
        except Exception:
            query_log = []
        _refiners[dataset_key] = QueryRefiner(vocabulary=vocab, query_log=query_log)
    return _refiners[dataset_key]


class RefineRequest(BaseModel):
    dataset: str = config.DEFAULT_DATASET
    query: str
    expand_synonyms: bool = True
    correct_spelling: bool = True


@app.get("/health")
def health():
    return {"status": "ok", "loaded": list(_refiners)}


@app.post("/refine")
def refine(req: RefineRequest):
    refiner = get_refiner(req.dataset)
    tokens = _pre.tokens(req.query)
    changes: dict[str, str] = {}
    corrected = tokens
    if req.correct_spelling:
        corrected, changes = refiner.correct_spelling(tokens)
    expanded = corrected
    if req.expand_synonyms:
        expanded = refiner.expand_synonyms(corrected, max_per_token=1)
    return {
        "original": req.query,
        "tokens": tokens,
        "corrected_tokens": corrected,
        "spelling_changes": changes,
        "expanded_tokens": expanded,
        "refined_query": " ".join(expanded),
    }


@app.get("/suggest")
def suggest(dataset: str = config.DEFAULT_DATASET, prefix: str = "", n: int = 5):
    refiner = get_refiner(dataset)
    return {"prefix": prefix, "suggestions": refiner.suggest(prefix, n=n)}
