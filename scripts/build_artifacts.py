"""Build all retrieval artifacts for a dataset (offline step).

Usage:
    python scripts/build_artifacts.py --dataset quora
    python scripts/build_artifacts.py --dataset quora --limit 5000   # quick dev
"""
from __future__ import annotations

import argparse

from irsys import config
from irsys.pipeline import build_artifacts
from irsys.preprocessing.text import PreprocessConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=config.DEFAULT_DATASET, choices=list(config.DATASETS))
    ap.add_argument("--limit", type=int, default=None, help="cap #docs (dev only)")
    ap.add_argument("--normalizer", default="lemmatize", choices=["lemmatize", "stem", "none"])
    ap.add_argument("--embedding-model", default=config.EMBEDDING_MODEL)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    pcfg = PreprocessConfig(normalizer=args.normalizer)
    meta = build_artifacts(
        dataset_key=args.dataset,
        limit=args.limit,
        preprocess_config=pcfg,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
    )
    print(meta)


if __name__ == "__main__":
    main()
