"""API Gateway — single entry point for the UI.

Routes requests to the independent services over REST (loose coupling). If a
downstream service is down, the gateway returns a clear error so the demo can
fall back to the direct-engine mode in the UI.

Run standalone:
    uvicorn services.gateway:app --port 8000
"""
from __future__ import annotations

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from irsys import config
from . import settings

app = FastAPI(title="IR · API Gateway")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

TIMEOUT = 60


def _forward(service: str, method: str, path: str, **kw):
    try:
        resp = requests.request(method, settings.url(service) + path, timeout=TIMEOUT, **kw)
    except requests.RequestException as e:
        raise HTTPException(502, f"{service} service unreachable: {e}")
    if resp.status_code >= 400:
        raise HTTPException(resp.status_code, resp.text)
    return resp.json()


@app.get("/health")
def health():
    out = {"gateway": "ok"}
    for svc in ("retrieval", "preprocessing", "refinement", "evaluation"):
        try:
            requests.get(settings.url(svc) + "/health", timeout=3)
            out[svc] = "ok"
        except requests.RequestException:
            out[svc] = "down"
    return out


@app.get("/datasets")
def datasets():
    return {"datasets": {k: v["label"] for k, v in config.DATASETS.items()}}


@app.post("/search")
def search(payload: dict):
    return _forward("retrieval", "POST", "/search", json=payload)


@app.post("/preprocess")
def preprocess(payload: dict):
    return _forward("preprocessing", "POST", "/preprocess", json=payload)


@app.post("/refine")
def refine(payload: dict):
    return _forward("refinement", "POST", "/refine", json=payload)


@app.get("/suggest")
def suggest(dataset: str = config.DEFAULT_DATASET, prefix: str = "", n: int = 5):
    return _forward("refinement", "GET", "/suggest",
                    params={"dataset": dataset, "prefix": prefix, "n": n})


@app.get("/report")
def report(dataset: str = config.DEFAULT_DATASET, tag: str = "baseline"):
    return _forward("evaluation", "GET", "/report", params={"dataset": dataset, "tag": tag})
