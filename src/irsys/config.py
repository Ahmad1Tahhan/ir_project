"""Central configuration: paths, dataset registry, model names.

Everything is resolved relative to the project root so the system is fully
portable / runnable offline (a hard requirement for the interview: no
internet is assumed in the lecture hall, so all caches live inside the repo).
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Keep the ir_datasets download cache *inside* the project for offline use.
IR_DATASETS_HOME = RAW_DIR / "ir_datasets"

for _p in (DATA_DIR, RAW_DIR, ARTIFACTS_DIR, MODELS_DIR, REPORTS_DIR, IR_DATASETS_HOME):
    _p.mkdir(parents=True, exist_ok=True)

# Point ir_datasets / HF at local caches before those libs are imported.
os.environ.setdefault("IR_DATASETS_HOME", str(IR_DATASETS_HOME))
os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(MODELS_DIR / "st"))

# Force fully-offline model loading: never contact the HF Hub at load time (all
# weights are already cached above). This removes the network round-trip + the
# "unauthenticated requests to the HF Hub" warning, and guarantees the demo runs
# with no internet (a hard requirement for the interview hall).
# To (re-)download a new model, comment these out, run once online, then restore.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# --- Models ----------------------------------------------------------------
# Small, fast, strong baseline; ~384-dim. Encodes 523K docs in minutes on a 4060.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Local LLM for the (optional) RAG feature — pick a small instruct model that
# fits in 8 GB VRAM. Resolved lazily so the rest of the system needs no LLM.
RAG_LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# --- Dataset registry ------------------------------------------------------
# Only datasets with >200K docs AND a qrels file are eligible (Antique is banned).
DATASETS: dict[str, dict] = {
    "quora": {
        "label": "BEIR / Quora",
        "irds_id": "beir/quora/test",   # 522,931 docs · 10,000 test queries · qrels
        "lang": "en",
        "doc_fields": ["text"],          # GenericDoc(doc_id, text)
        "query_fields": ["text"],        # GenericQuery(query_id, text)
    },
    # A second dataset slot (bonus). Fill in when chosen; pipeline is dataset-agnostic.
    # "wikir": {"label": "WikIR en1k", "irds_id": "wikir/en1k/test", ...},
}

DEFAULT_DATASET = "quora"


def artifacts_dir(dataset_key: str) -> Path:
    """Per-dataset artifact directory (indexes, embeddings, fitted models)."""
    d = ARTIFACTS_DIR / dataset_key
    d.mkdir(parents=True, exist_ok=True)
    return d
