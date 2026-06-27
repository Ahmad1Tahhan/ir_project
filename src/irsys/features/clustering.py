"""Document clustering (bonus feature).

KMeans over the cached document embeddings. Two uses:
  * Analysis/report — cluster sizes, silhouette score, 2-D PCA scatter chart.
  * Retrieval — restrict the candidate set to the query's nearest cluster(s),
    so the system can be evaluated "with clustering" vs. "without" (the demo
    requires each extra feature be testable independently).
"""
from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path

import numpy as np


class DocumentClusterer:
    def __init__(self, kmeans, labels: np.ndarray, doc_ids: list[str]):
        self.kmeans = kmeans
        self.labels = labels
        self.doc_ids = doc_ids
        self.n_clusters = int(kmeans.n_clusters)
        self._members: dict[int, list[int]] | None = None

    @classmethod
    def build(cls, embeddings: np.ndarray, doc_ids: list[str], n_clusters: int = 25,
              seed: int = 42) -> "DocumentClusterer":
        from sklearn.cluster import MiniBatchKMeans

        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed,
                             batch_size=4096, n_init="auto", max_iter=100)
        labels = km.fit_predict(embeddings)
        return cls(km, labels.astype(np.int32), doc_ids)

    # ----------------------------------------------------------- analysis
    def cluster_sizes(self) -> dict[int, int]:
        return dict(sorted(Counter(self.labels.tolist()).items()))

    def silhouette(self, embeddings: np.ndarray, sample: int = 5000, seed: int = 0) -> float:
        from sklearn.metrics import silhouette_score

        n = len(self.labels)
        if n > sample:
            rng = np.random.default_rng(seed)
            idx = rng.choice(n, size=sample, replace=False)
            return float(silhouette_score(embeddings[idx], self.labels[idx]))
        return float(silhouette_score(embeddings, self.labels))

    # ----------------------------------------------------------- retrieval
    def _members_map(self) -> dict[int, list[int]]:
        if self._members is None:
            m: dict[int, list[int]] = {c: [] for c in range(self.n_clusters)}
            for pos, c in enumerate(self.labels.tolist()):
                m[c].append(pos)
            self._members = m
        return self._members

    def nearest_clusters(self, query_vec: np.ndarray, n: int = 1) -> list[int]:
        d = np.linalg.norm(self.kmeans.cluster_centers_ - query_vec.reshape(1, -1), axis=1)
        return np.argsort(d)[:n].tolist()

    def candidate_doc_ids(self, query_vec: np.ndarray, n_clusters: int = 1) -> list[str]:
        members = self._members_map()
        out: list[str] = []
        for c in self.nearest_clusters(query_vec, n=n_clusters):
            out.extend(self.doc_ids[p] for p in members[c])
        return out

    # ----------------------------------------------------------- charts
    def plot_sizes(self, out_path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sizes = self.cluster_sizes()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar([str(k) for k in sizes], list(sizes.values()))
        ax.set_xlabel("cluster"); ax.set_ylabel("# documents")
        ax.set_title(f"Cluster sizes (k={self.n_clusters})")
        fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)

    def plot_scatter(self, embeddings: np.ndarray, out_path, sample: int = 4000, seed: int = 0):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        n = len(self.labels)
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=min(sample, n), replace=False)
        pts = PCA(n_components=2).fit_transform(embeddings[idx])
        fig, ax = plt.subplots(figsize=(8, 7))
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=self.labels[idx], cmap="tab20", s=6, alpha=0.6)
        ax.set_title("Document clusters (PCA 2-D)")
        fig.colorbar(sc, ax=ax, label="cluster")
        fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)

    # ----------------------------------------------------------- persist
    def save(self, directory):
        d = Path(directory); d.mkdir(parents=True, exist_ok=True)
        np.save(d / "cluster_labels.npy", self.labels)
        with open(d / "clusterer.pkl", "wb") as f:
            pickle.dump({"kmeans": self.kmeans, "doc_ids": self.doc_ids}, f)

    @classmethod
    def load(cls, directory):
        d = Path(directory)
        labels = np.load(d / "cluster_labels.npy")
        with open(d / "clusterer.pkl", "rb") as f:
            meta = pickle.load(f)
        return cls(meta["kmeans"], labels, meta["doc_ids"])
