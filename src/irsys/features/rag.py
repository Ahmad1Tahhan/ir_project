"""Retrieval-Augmented Generation (bonus feature).

Pipeline: retrieve top-k documents with the IR engine → build a grounded prompt
→ generate an answer with a LOCAL instruct LLM on the GPU (transformers). The
model is small enough to fit the 4060's VRAM and is cached locally so the demo
works offline.

Kept independent of the rest of the system: it only needs a RetrievalEngine to
fetch context, and degrades gracefully if the LLM cannot be loaded.
"""
from __future__ import annotations

from functools import lru_cache

from .. import config


@lru_cache(maxsize=1)
def _load_llm(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    return tok, model


def build_prompt(query: str, contexts: list[str]) -> list[dict]:
    ctx = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    system = (
        "You are a helpful assistant. Answer the user's question using ONLY the "
        "provided context passages. If the answer is not in the context, say you "
        "don't have enough information. Cite passages as [number]."
    )
    user = f"Context:\n{ctx}\n\nQuestion: {query}\n\nAnswer:"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


class RagPipeline:
    def __init__(self, engine, model_name: str = config.RAG_LLM_MODEL,
                 retrieval_method: str = "hybrid_serial"):
        self.engine = engine
        self.model_name = model_name
        self.retrieval_method = retrieval_method

    def retrieve(self, query: str, top_k: int = 5):
        return self.engine.search_with_text(self.retrieval_method, query, top_k=top_k)

    def generate(self, query: str, top_k: int = 5, max_new_tokens: int = 256) -> dict:
        import torch

        sources = self.retrieve(query, top_k=top_k)
        contexts = [s["text"] for s in sources if s["text"]]
        tok, model = _load_llm(self.model_name)
        messages = build_prompt(query, contexts)
        enc = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(model.device)
        prompt_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        answer = tok.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
        return {"answer": answer, "sources": sources}
