"""Build the bonus features (clustering + topic detection) from existing artifacts.

Reuses cached embeddings (clustering) and the TF-IDF matrix (topics) — no
re-encoding. Saves models + charts to data/artifacts/<dataset> and reports/.

Usage:
    python scripts/build_features.py --dataset quora --clustering --topics
    python scripts/build_features.py --dataset quora --clustering --n-clusters 30
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np

from irsys import config
from irsys.features.clustering import DocumentClusterer
from irsys.features.topics import TopicModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=config.DEFAULT_DATASET, choices=list(config.DATASETS))
    ap.add_argument("--clustering", action="store_true")
    ap.add_argument("--topics", action="store_true")
    ap.add_argument("--n-clusters", type=int, default=25)
    ap.add_argument("--n-topics", type=int, default=15)
    args = ap.parse_args()
    if not (args.clustering or args.topics):
        args.clustering = args.topics = True

    d = config.artifacts_dir(args.dataset)

    if args.clustering:
        print(f"[features] clustering (k={args.n_clusters}) ...")
        emb = np.load(d / "embeddings.npy")
        from irsys.indexing.vector_store import VectorStore
        doc_ids = VectorStore.load(d).doc_ids
        clu = DocumentClusterer.build(emb, doc_ids, n_clusters=args.n_clusters)
        clu.save(d)
        sizes = clu.cluster_sizes()
        sil = clu.silhouette(emb)
        print(f"  silhouette={sil:.4f}  sizes(min/max)={min(sizes.values())}/{max(sizes.values())}")
        clu.plot_sizes(config.REPORTS_DIR / f"clusters_{args.dataset}_sizes.png")
        clu.plot_scatter(emb, config.REPORTS_DIR / f"clusters_{args.dataset}_scatter.png")
        (config.REPORTS_DIR / f"clusters_{args.dataset}_silhouette.txt").write_text(
            f"silhouette={sil}\nsizes={sizes}\n", encoding="utf-8")

    if args.topics:
        print(f"[features] topic detection (NMF, K={args.n_topics}) ...")
        import scipy.sparse as sp
        doc_matrix = sp.load_npz(d / "tfidf_matrix.npz")
        with open(d / "tfidf_model.pkl", "rb") as f:
            vec = pickle.load(f)["vectorizer"]
        feats = vec.get_feature_names_out()
        tm = TopicModel.build_from_tfidf(doc_matrix, feats, n_topics=args.n_topics)
        tm.save(d)
        tw = tm.all_top_words(8)
        for t, words in tw.items():
            print(f"  topic {t}: {', '.join(words)}")
        tm.plot_topic_sizes(config.REPORTS_DIR / f"topics_{args.dataset}_sizes.png")
        tm.plot_top_words(config.REPORTS_DIR / f"topics_{args.dataset}_words.png")
        import json
        (config.REPORTS_DIR / f"topics_{args.dataset}_words.json").write_text(
            json.dumps(tw, indent=2), encoding="utf-8")

    print("[features] done.")


if __name__ == "__main__":
    main()
