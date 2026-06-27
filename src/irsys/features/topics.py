"""Topic detection (bonus feature) via NMF over the TF-IDF matrix.

Reuses the already-built TF-IDF representation (no recompute): NMF factorizes
the doc-term matrix into topics (term distributions) and per-document topic
weights. Produces the required topic charts and a per-document / per-query
dominant-topic lookup.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


class TopicModel:
    def __init__(self, nmf, feature_names: np.ndarray, doc_topics: np.ndarray):
        self.nmf = nmf
        self.feature_names = feature_names
        self.doc_topics = doc_topics            # (N, K) document-topic weights
        self.n_topics = int(nmf.n_components)

    @classmethod
    def build_from_tfidf(cls, doc_matrix, feature_names, n_topics: int = 15,
                         seed: int = 42, max_iter: int = 200) -> "TopicModel":
        from sklearn.decomposition import NMF

        nmf = NMF(n_components=n_topics, random_state=seed, init="nndsvda", max_iter=max_iter)
        W = nmf.fit_transform(doc_matrix)       # (N, K)
        return cls(nmf, np.asarray(feature_names), W.astype(np.float32))

    def top_words(self, topic: int, n: int = 10) -> list[str]:
        comp = self.nmf.components_[topic]
        idx = np.argsort(-comp)[:n]
        return self.feature_names[idx].tolist()

    def all_top_words(self, n: int = 10) -> dict[int, list[str]]:
        return {t: self.top_words(t, n) for t in range(self.n_topics)}

    def dominant_topic(self, doc_pos: int) -> int:
        return int(np.argmax(self.doc_topics[doc_pos]))

    def topic_sizes(self) -> dict[int, int]:
        from collections import Counter
        assign = np.argmax(self.doc_topics, axis=1)
        return dict(sorted(Counter(assign.tolist()).items()))

    # ----------------------------------------------------------- charts
    def plot_topic_sizes(self, out_path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sizes = self.topic_sizes()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar([str(k) for k in sizes], list(sizes.values()))
        ax.set_xlabel("topic"); ax.set_ylabel("# documents (dominant)")
        ax.set_title(f"Topic distribution (K={self.n_topics})")
        fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)

    def plot_top_words(self, out_path, n: int = 8):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        K = self.n_topics
        cols = 3
        rows = (K + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 2.6))
        axes = np.array(axes).reshape(-1)
        for t in range(K):
            comp = self.nmf.components_[t]
            idx = np.argsort(-comp)[:n][::-1]
            axes[t].barh(self.feature_names[idx], comp[idx])
            axes[t].set_title(f"Topic {t}", fontsize=9)
            axes[t].tick_params(labelsize=7)
        for j in range(K, len(axes)):
            axes[j].axis("off")
        fig.suptitle("Top words per topic")
        fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)

    # ----------------------------------------------------------- persist
    def save(self, directory):
        d = Path(directory); d.mkdir(parents=True, exist_ok=True)
        np.save(d / "doc_topics.npy", self.doc_topics)
        with open(d / "topic_model.pkl", "wb") as f:
            pickle.dump({"nmf": self.nmf, "feature_names": self.feature_names}, f)

    @classmethod
    def load(cls, directory):
        d = Path(directory)
        W = np.load(d / "doc_topics.npy")
        with open(d / "topic_model.pkl", "rb") as f:
            meta = pickle.load(f)
        return cls(meta["nmf"], meta["feature_names"], W)
