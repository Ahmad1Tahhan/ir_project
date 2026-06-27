# IR System — Information Retrieval Search Engine

A custom search engine over a large document collection (**BEIR/Quora**, ~523K
documents) built for the *Information Retrieval Systems 2026* course. Python only,
service-oriented (SOA), GPU-accelerated embeddings, with full IR evaluation.

## Features

**Core**
- **Preprocessing** — normalization, tokenization, stopword removal, lemmatization/stemming (same pipeline for documents *and* queries).
- **Representations**
  - VSM **TF-IDF** (cosine)
  - **BM25** over a custom **inverted index** — `k1`, `b` adjustable live per query
  - **Embeddings** (sentence-transformers `all-MiniLM-L6-v2`, GPU) + **FAISS** vector store
  - **Hybrid — Serial** (BM25 → embedding rerank) and **Hybrid — Parallel** (RRF / weighted-sum fusion)
- **Indexing** — inverted index (sparse) + FAISS (dense)
- **Query refinement** — spelling correction, WordNet synonym expansion, suggestions, history weighting
- **Matching & ranking** — per-representation similarity (cosine, BM25 score, fused)
- **SOA** — independent FastAPI services behind an API gateway
- **Evaluation** — MAP, Recall, Precision@10, nDCG over **all** qrels queries; before/after extra features; charts
- **UI** — Streamlit (dataset picker, method picker, live BM25 params, hybrid mode, clustering toggle, shows original docs + IDs)

**Bonus features** (independently testable)
- FAISS **vector store** · document **clustering** (KMeans + cluster-restricted search) · **topic detection** (NMF) · **RAG** chat (local GPU LLM)

## Architecture (SOA)

```
            ┌─────────────┐
   UI  ───▶ │ API Gateway │ ─┬─▶ Retrieval Service   (TF-IDF/BM25/embeddings/hybrid + clustering)
(Streamlit) │   :8000     │  ├─▶ Preprocessing Service
            └─────────────┘  ├─▶ Query Refinement Service
                             ├─▶ Ranking & Evaluation Service
                             └─▶ RAG Service (optional, local LLM)
```
All services import the shared core library `irsys` (`src/irsys/`).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate                                  # Windows
pip install torch --index-url https://download.pytorch.org/whl/cu124   # GPU torch
pip install -r requirements.txt
pip install -e .
```

## Build artifacts (offline, one-time)

```bash
python scripts/build_artifacts.py --dataset quora       # preprocess + TF-IDF + BM25 + embeddings (~90s on a 4060)
python scripts/build_features.py  --dataset quora        # clustering + topic detection (+ charts)
```

## Evaluate

```bash
python scripts/evaluate.py --dataset quora --tag baseline
# -> prints NUMBER OF QUERIES USED, writes reports/eval_quora_baseline.json + .png
```
Or open `notebooks/evaluation.ipynb` (prints the query count, shows the charts).

## Run the system

```bash
./scripts/start_all.ps1        # starts all services + the Streamlit UI
# Gateway health: http://127.0.0.1:8000/health   ·   UI: http://localhost:8501
```
The UI also has a **Direct** backend mode that loads the engine in-process — no
servers required (handy offline fallback for the demo).

## Layout

```
src/irsys/        core library (preprocessing, representation, indexing,
                  retrieval, refinement, evaluation, features, pipeline)
services/         FastAPI microservices + gateway (SOA)
ui/app.py         Streamlit UI
scripts/          build_artifacts · build_features · evaluate · smoke_test · start_all
notebooks/        evaluation.ipynb
data/             raw/ (ir_datasets cache) · artifacts/<dataset>/ (indexes, embeddings)
reports/          metrics JSON + charts
```

## Notes
- Everything runs fully **offline** after the build step (caches live inside the repo).
- Antique dataset is **not** used (banned). Dataset has a qrels file (required for evaluation).
