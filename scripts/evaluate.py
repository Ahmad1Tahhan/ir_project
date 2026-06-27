"""Full evaluation: MAP, Recall, P@10, nDCG per representation, over ALL qrels.

Prints the number of queries used (required to be shown in the interview), saves
a metrics JSON, and renders a comparison bar chart.

Usage:
    python scripts/evaluate.py --dataset quora
    python scripts/evaluate.py --dataset quora --methods bm25 embedding hybrid_serial
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from irsys import config
from irsys.data import loaders
from irsys.evaluation import evaluate_run
from irsys.pipeline import METHODS, RetrievalEngine


def build_run(eng: RetrievalEngine, method: str, queries: dict[str, str], top_k: int) -> dict:
    """Return {qid: [doc_id,...]} for a method, batching where it helps."""
    qids = list(queries)
    run: dict[str, list[str]] = {}

    if method in ("embedding", "embedding_clustered"):
        # Batch-encode every query on GPU once, then search each.
        from irsys.representation.embeddings import encode_texts

        clean = [" ".join(eng.pre.tokens(queries[q])) for q in qids]
        qvecs = encode_texts(clean, model_name=eng.embedding.model_name,
                             batch_size=256, show_progress=True)
        clustered = method == "embedding_clustered"
        for qid, qv in zip(qids, qvecs):
            if clustered:
                cand = eng.clusterer.candidate_doc_ids(qv, n_clusters=1)
                # rerank candidate cluster by cosine to the (already-clean) query
                pos = [eng.embedding.id_to_pos[c] for c in cand]
                sims = eng.embedding.embeddings[pos] @ qv
                order = np.argsort(-sims)[:top_k]
                run[qid] = [cand[o] for o in order]
            else:
                run[qid] = [d for d, _ in eng.embedding.store.search(qv, top_k=top_k)]
        return run

    for i, qid in enumerate(qids):
        run[qid] = [d for d, _ in eng.search(method, queries[qid], top_k=top_k)]
        if (i + 1) % 1000 == 0:
            print(f"  [{method}] {i+1}/{len(qids)}")
    return run


def chart(results: dict, out_path, dataset_key: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = list(results)
    metric_keys = [k for k in results[methods[0]] if k != "num_queries"]
    x = np.arange(len(metric_keys))
    width = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, m in enumerate(methods):
        vals = [results[m][k] for k in metric_keys]
        ax.bar(x + i * width, vals, width, label=m)
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(metric_keys)
    ax.set_ylabel("score")
    ax.set_title(f"IR evaluation — {dataset_key}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"[eval] chart -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=config.DEFAULT_DATASET, choices=list(config.DATASETS))
    ap.add_argument("--methods", nargs="*", default=METHODS,
                    choices=METHODS + ["embedding_clustered"])
    ap.add_argument("--top-k", type=int, default=100, help="ranking depth fed to metrics")
    ap.add_argument("--tag", default="baseline", help="label for this run (e.g. baseline / with_features)")
    args = ap.parse_args()

    eng = RetrievalEngine.load(args.dataset)
    queries = loaders.load_queries(args.dataset)
    qrels = loaders.load_qrels(args.dataset)
    # Use only queries that have at least one relevant doc in qrels — ALL of them.
    queries = {q: queries[q] for q in qrels if q in queries and any(r > 0 for r in qrels[q].values())}
    print(f"\n[eval] dataset={args.dataset}  NUMBER OF QUERIES USED = {len(queries)}\n")

    results = {}
    for method in args.methods:
        t = time.time()
        print(f"[eval] running {method} ...")
        run = build_run(eng, method, queries, args.top_k)
        ev = evaluate_run(run, qrels)
        results[method] = ev["metrics"]
        print(f"  -> {method}: {ev['metrics']}  ({time.time()-t:.1f}s)")

    out = config.REPORTS_DIR / f"eval_{args.dataset}_{args.tag}.json"
    payload = {"dataset": args.dataset, "tag": args.tag,
               "num_queries": len(queries), "results": results}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[eval] metrics -> {out}")
    chart(results, config.REPORTS_DIR / f"eval_{args.dataset}_{args.tag}.png", args.dataset)


if __name__ == "__main__":
    main()
