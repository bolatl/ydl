"""RAG retrieval. The single source of *facts* for the bot."""
from __future__ import annotations

from functools import lru_cache

import chromadb

from core import config
from core.embeddings import embed_one


@lru_cache(maxsize=1)
def _collection():
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_collection(config.CHROMA_COLLECTION)


def search_context(question: str, n_results: int = config.N_RESULTS) -> list[dict]:
    """Return relevant chunks as [{text, url, distance}].

    Chunks farther than config.MAX_DISTANCE are dropped, so an off-topic question
    yields an empty list and the caller can refuse instead of hallucinating.
    """
    question = (question or "").strip()
    if not question:
        return []

    try:
        collection = _collection()
    except Exception as exc:  # collection missing -> nothing indexed yet
        raise RuntimeError(
            "Chroma collection not found. Run `python build_index.py` first."
        ) from exc

    res = collection.query(
        query_embeddings=[embed_one(question)],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    hits: list[dict] = []
    for text, meta, dist in zip(docs, metas, dists):
        if dist is None or dist <= config.MAX_DISTANCE:
            hits.append({"text": text, "url": (meta or {}).get("url", ""), "distance": dist})
    return hits
