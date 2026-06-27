"""Preprocessing Service — exposes the shared text pipeline over REST.

Run standalone:
    uvicorn services.preprocessing_service:app --port 8002
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from irsys.preprocessing.text import PreprocessConfig, TextPreprocessor

app = FastAPI(title="IR · Preprocessing Service")
_cache: dict[str, TextPreprocessor] = {}


def get_pre(normalizer: str) -> TextPreprocessor:
    if normalizer not in _cache:
        _cache[normalizer] = TextPreprocessor(PreprocessConfig(normalizer=normalizer))
    return _cache[normalizer]


class PreprocessRequest(BaseModel):
    text: str
    normalizer: str = "lemmatize"   # "lemmatize" | "stem" | "none"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/preprocess")
def preprocess(req: PreprocessRequest):
    pre = get_pre(req.normalizer)
    tokens = pre.tokens(req.text)
    return {"tokens": tokens, "clean": " ".join(tokens), "num_tokens": len(tokens)}
