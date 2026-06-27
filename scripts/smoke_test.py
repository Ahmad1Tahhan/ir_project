"""Quick end-to-end validation on a small slice (no full build).

Downloads/uses the dataset cache, builds artifacts for a few thousand docs,
runs every retrieval method, and runs the evaluator. Purpose: catch code bugs
before the expensive full build.
"""
from __future__ import annotations

import time

from irsys import config
from irsys.data import loaders
from irsys.evaluation import evaluate_run
from irsys.pipeline import RetrievalEngine, build_artifacts
from irsys.preprocessing.text import PreprocessConfig

DATASET = "quora"
LIMIT = 4000


def main():
    t = time.time()
    build_artifacts(DATASET, limit=LIMIT, preprocess_config=PreprocessConfig(), batch_size=256)
    eng = RetrievalEngine.load(DATASET)
    print(f"\nengine loaded; vocab={len(eng.vocabulary):,}")

    q = "how do i learn python programming"
    for method in ["tfidf", "bm25", "embedding", "hybrid_serial", "hybrid_parallel"]:
        res = eng.search_with_text(method, q, top_k=3)
        print(f"\n=== {method} ===")
        for r in res:
            print(f"  {r['doc_id']:>10}  {r['score']:.4f}  {r['text'][:70]!r}")

    # mini-eval over a few queries (recall will be low: only LIMIT docs indexed)
    queries = loaders.load_queries(DATASET)
    qrels = loaders.load_qrels(DATASET)
    some_qids = list(qrels)[:50]
    run = {}
    for qid in some_qids:
        run[qid] = [d for d, _ in eng.search("bm25", queries[qid], top_k=100)]
    ev = evaluate_run(run, {q: qrels[q] for q in some_qids})
    print("\nmini-eval (bm25, 50 queries, subset corpus):", ev["metrics"])
    print(f"\nsmoke test finished in {time.time()-t:.1f}s")


if __name__ == "__main__":
    main()
