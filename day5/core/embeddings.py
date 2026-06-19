"""Embedding seam. Default: course-issued OpenAI-compatible embeddings endpoint.

If the API does not serve embeddings, set EMBEDDING_MODEL to a sentence-transformers
id (e.g. "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") and this
module transparently falls back to a local model. No other file needs to change.
"""
from __future__ import annotations

from functools import lru_cache

from core import config

_BATCH = 64


def _use_local() -> bool:
    """Heuristic: a slash in the model id means a HF/sentence-transformers model."""
    return "/" in config.EMBEDDING_MODEL


@lru_cache(maxsize=1)
def _openai_client():
    from openai import OpenAI

    if not (config.EMBEDDING_API_KEY and config.EMBEDDING_MODEL):
        raise RuntimeError(
            "Missing EMBEDDING_API_KEY / EMBEDDING_MODEL in .env. "
            "On this provider the embedding service uses a separate key."
        )
    return OpenAI(
        api_key=config.EMBEDDING_API_KEY,
        base_url=config.EMBEDDING_BASE_URL or None,
    )


@lru_cache(maxsize=1)
def _local_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBEDDING_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts -> list of float vectors (one per input)."""
    if not texts:
        return []

    if _use_local():
        model = _local_model()
        return [vec.tolist() for vec in model.encode(texts, show_progress_bar=False)]

    client = _openai_client()
    vectors: list[list[float]] = []
    for i in range(0, len(texts), _BATCH):
        batch = texts[i : i + _BATCH]
        resp = client.embeddings.create(model=config.EMBEDDING_MODEL, input=batch)
        vectors.extend(item.embedding for item in resp.data)
    return vectors


def embed_one(text: str) -> list[float]:
    return embed([text])[0]
