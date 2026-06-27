"""Dataset loading via ir_datasets.

Loads documents, queries, and qrels for a registered dataset. Documents are
kept in their ORIGINAL (uncleaned) form here — the UI must show the original
text + doc_id. Cleaning happens later in the representation pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from .. import config


def _dataset(dataset_key: str):
    import ir_datasets

    spec = config.DATASETS[dataset_key]
    return ir_datasets.load(spec["irds_id"]), spec


def _doc_text(doc, fields) -> str:
    parts = []
    for f in fields:
        v = getattr(doc, f, None)
        if v:
            parts.append(str(v))
    return " ".join(parts).strip()


@dataclass
class Corpus:
    """In-memory corpus: parallel lists + id<->position lookup."""

    dataset_key: str
    doc_ids: list[str]
    raw_texts: list[str]          # original, uncleaned (shown in UI)
    id_to_pos: dict[str, int]

    def __len__(self) -> int:
        return len(self.doc_ids)

    def text(self, doc_id: str) -> str:
        return self.raw_texts[self.id_to_pos[doc_id]]


def iter_docs(dataset_key: str) -> Iterator[tuple[str, str]]:
    """Stream (doc_id, raw_text) without holding everything in memory."""
    ds, spec = _dataset(dataset_key)
    fields = spec["doc_fields"]
    for doc in ds.docs_iter():
        yield doc.doc_id, _doc_text(doc, fields)


def load_corpus(dataset_key: str, limit: int | None = None) -> Corpus:
    """Load the full corpus into memory. ``limit`` is for quick dev runs only;
    evaluation/production must use the full corpus."""
    doc_ids: list[str] = []
    raw_texts: list[str] = []
    for i, (doc_id, text) in enumerate(iter_docs(dataset_key)):
        if limit is not None and i >= limit:
            break
        doc_ids.append(doc_id)
        raw_texts.append(text)
    id_to_pos = {d: i for i, d in enumerate(doc_ids)}
    return Corpus(dataset_key, doc_ids, raw_texts, id_to_pos)


def load_queries(dataset_key: str) -> dict[str, str]:
    """Return {query_id: raw_query_text} for ALL queries (eval uses all of them)."""
    ds, spec = _dataset(dataset_key)
    qfields = spec["query_fields"]
    return {q.query_id: _doc_text(q, qfields) for q in ds.queries_iter()}


def get_docs_store(dataset_key: str):
    """Random-access store for original document text (used by the UI to show
    the uncleaned document for a given doc_id without loading the whole corpus)."""
    ds, _ = _dataset(dataset_key)
    return ds.docs_store()


def fetch_original(dataset_key: str, doc_ids: list[str]) -> dict[str, str]:
    """Return {doc_id: original_text} for the given ids (for result display)."""
    store = get_docs_store(dataset_key)
    spec = config.DATASETS[dataset_key]
    fields = spec["doc_fields"]
    out: dict[str, str] = {}
    for did in doc_ids:
        try:
            doc = store.get(did)
            out[did] = _doc_text(doc, fields)
        except Exception:
            out[did] = ""
    return out


def load_qrels(dataset_key: str) -> dict[str, dict[str, int]]:
    """Return {query_id: {doc_id: relevance}} — the ground truth for evaluation."""
    ds, _ = _dataset(dataset_key)
    qrels: dict[str, dict[str, int]] = {}
    for qr in ds.qrels_iter():
        qrels.setdefault(qr.query_id, {})[qr.doc_id] = int(qr.relevance)
    return qrels
