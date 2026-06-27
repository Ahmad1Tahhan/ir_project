"""Fusion methods for the PARALLEL hybrid representation.

When several representations run independently, their result lists must be
merged into one ranking. Two standard methods are provided:

  * Reciprocal Rank Fusion (RRF) — rank-based, robust to incompatible score
    scales (BM25 vs cosine). score(d) = Σ 1 / (k + rank_i(d)).
  * Weighted-sum — min-max normalize each list's scores, then weighted sum.
"""
from __future__ import annotations

from typing import Sequence

Ranked = Sequence[tuple[str, float]]


def reciprocal_rank_fusion(
    result_lists: Sequence[Ranked], weights: Sequence[float] | None = None, k: int = 60
) -> list[tuple[str, float]]:
    weights = weights or [1.0] * len(result_lists)
    fused: dict[str, float] = {}
    for w, results in zip(weights, result_lists):
        for rank, (doc_id, _score) in enumerate(results, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + w * (1.0 / (k + rank))
    return sorted(fused.items(), key=lambda x: -x[1])


def _minmax(results: Ranked) -> dict[str, float]:
    if not results:
        return {}
    scores = [s for _, s in results]
    lo, hi = min(scores), max(scores)
    rng = hi - lo
    if rng <= 0:
        return {d: 1.0 for d, _ in results}
    return {d: (s - lo) / rng for d, s in results}


def weighted_sum_fusion(
    result_lists: Sequence[Ranked], weights: Sequence[float] | None = None
) -> list[tuple[str, float]]:
    weights = weights or [1.0] * len(result_lists)
    fused: dict[str, float] = {}
    for w, results in zip(weights, result_lists):
        norm = _minmax(results)
        for doc_id, s in norm.items():
            fused[doc_id] = fused.get(doc_id, 0.0) + w * s
    return sorted(fused.items(), key=lambda x: -x[1])


FUSION_METHODS = {
    "rrf": reciprocal_rank_fusion,
    "weighted": weighted_sum_fusion,
}
