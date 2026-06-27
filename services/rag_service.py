"""RAG Service — retrieval-augmented chat over the corpus with a local LLM.

Run standalone:
    uvicorn services.rag_service:app --port 8005
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from irsys import config
from irsys.pipeline import RetrievalEngine

app = FastAPI(title="IR · RAG Service")
_pipelines: dict[str, object] = {}


def get_pipeline(dataset_key: str):
    if dataset_key not in _pipelines:
        from irsys.features.rag import RagPipeline
        eng = RetrievalEngine.load(dataset_key)
        _pipelines[dataset_key] = RagPipeline(eng)
    return _pipelines[dataset_key]


class ChatRequest(BaseModel):
    dataset: str = config.DEFAULT_DATASET
    query: str
    top_k: int = 5
    max_new_tokens: int = 256


@app.get("/health")
def health():
    return {"status": "ok", "model": config.RAG_LLM_MODEL}


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        pipe = get_pipeline(req.dataset)
        return pipe.generate(req.query, top_k=req.top_k, max_new_tokens=req.max_new_tokens)
    except Exception as e:
        raise HTTPException(500, f"RAG failed: {e}")
