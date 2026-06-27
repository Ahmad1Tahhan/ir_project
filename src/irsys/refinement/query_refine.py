"""Query refinement (independently testable service).

Three complementary techniques to increase result quality:
  * Spelling correction — fix out-of-vocabulary tokens against the index vocab
    (edit-distance via difflib).
  * Synonym expansion — add WordNet synonyms so semantically-equivalent docs
    are reachable by lexical models (TF-IDF / BM25).
  * Suggestion / autocomplete — propose full queries from the dataset's own
    query log (or frequent vocab terms) given a prefix.

A simple per-session search history can additionally up-weight recently used
terms (personalization-lite), kept optional so the feature is testable alone.
"""
from __future__ import annotations

import difflib
from collections import Counter


class QueryRefiner:
    def __init__(self, vocabulary: set[str] | None = None, query_log: list[str] | None = None):
        self.vocabulary = vocabulary or set()
        self.query_log = query_log or []
        self._suggest_pool = sorted(set(self.query_log))

    # ----------------------------------------------------------- spelling
    def correct_spelling(self, tokens: list[str], cutoff: float = 0.85) -> tuple[list[str], dict[str, str]]:
        if not self.vocabulary:
            return tokens, {}
        corrected, changes = [], {}
        vocab_list = self._vocab_list()
        for tok in tokens:
            if tok in self.vocabulary or tok.isdigit():
                corrected.append(tok)
                continue
            match = difflib.get_close_matches(tok, vocab_list, n=1, cutoff=cutoff)
            if match:
                corrected.append(match[0])
                changes[tok] = match[0]
            else:
                corrected.append(tok)
        return corrected, changes

    def _vocab_list(self) -> list[str]:
        if not hasattr(self, "_vlist"):
            self._vlist = list(self.vocabulary)
        return self._vlist

    # ----------------------------------------------------------- synonyms
    def expand_synonyms(self, tokens: list[str], max_per_token: int = 1) -> list[str]:
        from nltk.corpus import wordnet as wn

        from ..preprocessing.text import ensure_nltk_data

        ensure_nltk_data()
        extra: list[str] = []
        for tok in tokens:
            seen = set()
            for syn in wn.synsets(tok):
                for lemma in syn.lemmas():
                    w = lemma.name().replace("_", " ").lower()
                    if w != tok and w not in seen and " " not in w:
                        seen.add(w)
                    if len(seen) >= max_per_token:
                        break
                if len(seen) >= max_per_token:
                    break
            extra.extend(seen)
        return tokens + extra

    # --------------------------------------------------------- suggestion
    def suggest(self, prefix: str, n: int = 5) -> list[str]:
        prefix = prefix.lower().strip()
        if not prefix:
            return self._suggest_pool[:n]
        starts = [q for q in self._suggest_pool if q.lower().startswith(prefix)]
        if len(starts) < n:
            contains = [q for q in self._suggest_pool if prefix in q.lower() and q not in starts]
            starts += contains
        return starts[:n]

    # ------------------------------------------------- history weighting
    @staticmethod
    def weight_with_history(tokens: list[str], history: list[str], repeat: int = 1) -> list[str]:
        """Up-weight query terms that appear in the user's recent history by
        duplicating them (boosts their TF in lexical models)."""
        if not history:
            return tokens
        hist_terms = set(Counter(" ".join(history).lower().split()))
        boosted = list(tokens)
        for tok in tokens:
            if tok in hist_terms:
                boosted.extend([tok] * repeat)
        return boosted
