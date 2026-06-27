"""Streamlit UI for the IR system.

Tabs: Search · Evaluation · RAG chat. Two backends, switchable in the sidebar:
  * Gateway (SOA)  — calls the API gateway / microservices over REST.
  * Direct         — loads the RetrievalEngine in-process (robust, no servers
                     needed; ideal as an offline fallback in the interview).

Run:
    streamlit run ui/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests
import streamlit as st

# Make the local package importable when run via `streamlit run`.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from irsys import config                       # noqa: E402
from services import settings                  # noqa: E402

st.set_page_config(page_title="IR Search System", page_icon="🔎", layout="wide")

METHOD_LABELS = {
    "bm25": "BM25 (probabilistic)",
    "tfidf": "TF-IDF (VSM)",
    "embedding": "Embeddings (semantic)",
    "hybrid_serial": "Hybrid — Serial (BM25 → rerank)",
    "hybrid_parallel": "Hybrid — Parallel (fusion)",
}


# --------------------------------------------------------------- direct engine
@st.cache_resource(show_spinner="Loading retrieval engine (indexes + embeddings)…")
def load_engine(dataset_key: str):
    from irsys.pipeline import RetrievalEngine
    return RetrievalEngine.load(dataset_key)


def gw_search(payload: dict) -> dict:
    r = requests.post(settings.url("gateway") + "/search", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def do_search(backend: str, dataset: str, method: str, query: str, top_k: int,
              k1: float, b: float, fusion: str, candidate_k: int,
              use_clustering: bool, n_clusters: int) -> dict:
    if backend == "Gateway (SOA)":
        return gw_search({
            "dataset": dataset, "method": method, "query": query, "top_k": top_k,
            "k1": k1, "b": b, "fusion": fusion, "candidate_k": candidate_k,
            "use_clustering": use_clustering, "n_clusters": n_clusters,
        })
    # Direct mode
    eng = load_engine(dataset)
    import time
    t0 = time.time()
    if use_clustering:
        ranked = eng.search_clustered(query, top_k=top_k, n_clusters=n_clusters)
        from irsys.data import loaders
        originals = loaders.fetch_original(dataset, [d for d, _ in ranked])
        items = [{"doc_id": d, "score": float(s), "text": originals.get(d, "")} for d, s in ranked]
    else:
        items = eng.search_with_text(method, query, top_k=top_k,
                                     k1=k1, b=b, fusion=fusion, candidate_k=candidate_k)
    return {"results": items, "took_ms": round((time.time() - t0) * 1000, 1),
            "num_results": len(items)}


# ------------------------------------------------------------------- sidebar
st.sidebar.title("🔎 IR System")
backend = st.sidebar.radio("Backend", ["Direct", "Gateway (SOA)"],
                           help="Direct loads the engine in-process. Gateway uses the microservices.")
dataset = st.sidebar.selectbox("Dataset", list(config.DATASETS),
                               format_func=lambda k: config.DATASETS[k]["label"])
st.sidebar.caption("Pick the dataset BEFORE querying (per the spec).")

if backend == "Gateway (SOA)":
    try:
        h = requests.get(settings.url("gateway") + "/health", timeout=3).json()
        st.sidebar.success("gateway: " + ", ".join(f"{k}={v}" for k, v in h.items()))
    except requests.RequestException:
        st.sidebar.error("Gateway not reachable — start services or use Direct.")

tab_search, tab_eval, tab_rag = st.tabs(["Search", "Evaluation", "RAG chat"])

# --------------------------------------------------------------------- SEARCH
with tab_search:
    st.subheader("Search")
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Query", value="how do i learn python programming")
    with col2:
        top_k = st.number_input("Top-K", 1, 100, 10)

    method = st.selectbox("Representation / method", list(METHOD_LABELS),
                          format_func=lambda m: METHOD_LABELS[m])

    mode = st.radio("Execution mode", ["Basic only", "Basic + extra (clustering)"],
                    horizontal=True)
    use_clustering = mode.startswith("Basic +")

    with st.expander("Parameters", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            k1 = st.slider("BM25 k1", 0.1, 3.0, 1.5, 0.1,
                           help="Term-frequency saturation (BM25).")
            b = st.slider("BM25 b", 0.0, 1.0, 0.75, 0.05,
                          help="Length normalization (BM25).")
        with c2:
            fusion = st.selectbox("Parallel fusion", ["rrf", "weighted"])
            candidate_k = st.slider("Candidate pool (hybrid)", 20, 500, 100, 10)
        with c3:
            n_clusters = st.slider("# nearest clusters (clustering mode)", 1, 5, 1)

    # query refinement helper (best-effort)
    if st.checkbox("Show query refinement suggestions"):
        try:
            if backend == "Gateway (SOA)":
                ref = requests.post(settings.url("gateway") + "/refine",
                                    json={"dataset": dataset, "query": query}, timeout=30).json()
            else:
                import pickle
                from irsys.refinement import QueryRefiner
                from irsys.preprocessing.text import PreprocessConfig, TextPreprocessor
                with open(config.artifacts_dir(dataset) / "vocabulary.pkl", "rb") as f:
                    vocab = pickle.load(f)
                pre = TextPreprocessor(PreprocessConfig())
                refiner = QueryRefiner(vocabulary=vocab)
                toks = pre.tokens(query)
                corrected, changes = refiner.correct_spelling(toks)
                ref = {"spelling_changes": changes,
                       "refined_query": " ".join(refiner.expand_synonyms(corrected))}
            if ref.get("spelling_changes"):
                st.info(f"Spelling: {ref['spelling_changes']}")
            st.caption(f"Refined query → `{ref.get('refined_query','')}`")
        except Exception as e:
            st.warning(f"Refinement unavailable: {e}")

    if st.button("Search", type="primary"):
        with st.spinner("Searching…"):
            try:
                out = do_search(backend, dataset, method, query, top_k, k1, b,
                                fusion, candidate_k, use_clustering, n_clusters)
            except Exception as e:
                st.error(f"Search failed: {e}")
                out = None
        if out:
            st.caption(f"{out['num_results']} results · {out['took_ms']} ms · "
                       f"{'clustering ON' if use_clustering else method}")
            for i, r in enumerate(out["results"], 1):
                st.markdown(f"**{i}. doc_id `{r['doc_id']}`** · score `{r['score']:.4f}`")
                st.write(r["text"] or "_(empty)_")
                st.divider()

# ----------------------------------------------------------------- EVALUATION
with tab_eval:
    st.subheader("Evaluation")
    tag = st.text_input("Report tag", value="baseline")
    if st.button("Load report"):
        try:
            if backend == "Gateway (SOA)":
                rep = requests.get(settings.url("gateway") + "/report",
                                   params={"dataset": dataset, "tag": tag}, timeout=30).json()
            else:
                import json
                p = config.REPORTS_DIR / f"eval_{dataset}_{tag}.json"
                rep = json.loads(p.read_text(encoding="utf-8"))
            st.success(f"Number of queries used: {rep['num_queries']}")
            import pandas as pd
            df = pd.DataFrame(rep["results"]).T
            st.dataframe(df)
            st.bar_chart(df[[c for c in df.columns if c != "num_queries"]])
            png = config.REPORTS_DIR / f"eval_{dataset}_{tag}.png"
            if png.exists():
                st.image(str(png))
        except Exception as e:
            st.error(f"Could not load report: {e}")

# ------------------------------------------------------------------- RAG chat
with tab_rag:
    st.subheader("RAG chat (retrieval-augmented)")
    st.caption("Answers are generated from retrieved documents using a local LLM.")
    if "rag_history" not in st.session_state:
        st.session_state.rag_history = []
    for role, msg in st.session_state.rag_history:
        with st.chat_message(role):
            st.write(msg)
    prompt = st.chat_input("Ask a question…")
    if prompt:
        st.session_state.rag_history.append(("user", prompt))
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"), st.spinner("Thinking…"):
            try:
                r = requests.post(settings.url("rag") + "/chat",
                                  json={"dataset": dataset, "query": prompt, "top_k": 5},
                                  timeout=180)
                r.raise_for_status()
                data = r.json()
                answer = data["answer"]
                st.write(answer)
                with st.expander("Sources"):
                    for s in data.get("sources", []):
                        st.markdown(f"- `{s['doc_id']}` — {s['text'][:120]}")
            except Exception as e:
                answer = f"RAG service unavailable: {e}\n\nStart it with the launcher (rag service)."
                st.warning(answer)
        st.session_state.rag_history.append(("assistant", answer))
