"""IR evaluation metrics — implemented from scratch.

Metrics required by the spec (computed per representation, per dataset,
before/after the extra features): MAP, Recall, Precision@10, nDCG.

Relevance is taken from qrels as {query_id: {doc_id: relevance}}. A document
is "relevant" when relevance > 0. nDCG uses graded relevance as gain.

A *run* is {query_id: [doc_id, ...]} ranked best-first.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

Run = Mapping[str, Sequence[str]]
Qrels = Mapping[str, Mapping[str, int]]


def average_precision(ranking: Sequence[str], rel: Mapping[str, int]) -> float:
    n_rel = sum(1 for r in rel.values() if r > 0)
    if n_rel == 0:
        return 0.0
    hits = 0
    score = 0.0
    for i, doc_id in enumerate(ranking, start=1):
        if rel.get(doc_id, 0) > 0:
            hits += 1
            score += hits / i
    return score / n_rel


def precision_at_k(ranking: Sequence[str], rel: Mapping[str, int], k: int = 10) -> float:
    if k <= 0:
        return 0.0
    top = ranking[:k]
    hits = sum(1 for d in top if rel.get(d, 0) > 0)
    return hits / k


def recall_at_k(ranking: Sequence[str], rel: Mapping[str, int], k: int = 1000) -> float:
    n_rel = sum(1 for r in rel.values() if r > 0)
    if n_rel == 0:
        return 0.0
    top = ranking[:k]
    hits = sum(1 for d in top if rel.get(d, 0) > 0)
    return hits / n_rel


def _dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranking: Sequence[str], rel: Mapping[str, int], k: int = 10) -> float:
    gains = [max(rel.get(d, 0), 0) for d in ranking[:k]]
    dcg = _dcg(gains)
    ideal = sorted((r for r in rel.values() if r > 0), reverse=True)[:k]
    idcg = _dcg(ideal)
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_run(
    run: Run,
    qrels: Qrels,
    p_k: int = 10,
    ndcg_k: int = 10,
    recall_k: int = 1000,
) -> dict:
    """Macro-average each metric over all queries that have qrels.

    Returns a dict with the mean metrics plus ``num_queries`` (the count the
    interview requires us to print) and per-query breakdowns.
    """
    per_query: dict[str, dict] = {}
    # Evaluate over every query present in qrels (use ALL of them, per the rules).
    eval_qids = [q for q in qrels if any(r > 0 for r in qrels[q].values())]

    sums = {"MAP": 0.0, f"P@{p_k}": 0.0, f"nDCG@{ndcg_k}": 0.0, f"Recall@{recall_k}": 0.0}
    for qid in eval_qids:
        rel = qrels[qid]
        ranking = list(run.get(qid, []))
        ap = average_precision(ranking, rel)
        p = precision_at_k(ranking, rel, p_k)
        nd = ndcg_at_k(ranking, rel, ndcg_k)
        rc = recall_at_k(ranking, rel, recall_k)
        per_query[qid] = {"AP": ap, f"P@{p_k}": p, f"nDCG@{ndcg_k}": nd, f"Recall@{recall_k}": rc}
        sums["MAP"] += ap
        sums[f"P@{p_k}"] += p
        sums[f"nDCG@{ndcg_k}"] += nd
        sums[f"Recall@{recall_k}"] += rc

    n = len(eval_qids) or 1
    means = {k: v / n for k, v in sums.items()}
    means["num_queries"] = len(eval_qids)
    return {"metrics": means, "per_query": per_query}
