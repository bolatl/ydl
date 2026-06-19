"""Chunk raw pages, embed them, and build the local Chroma index.

Usage:
    python build_index.py
"""
from __future__ import annotations

import json

import chromadb

from core import config
from core.embeddings import embed


def load_pages() -> list[dict]:
    if not config.RAW_PAGES_PATH.exists():
        raise FileNotFoundError(
            f"{config.RAW_PAGES_PATH} not found. Run `python scrape.py` first."
        )
    with open(config.RAW_PAGES_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Character-based sliding window with overlap. Tries to break on whitespace."""
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        # Prefer to cut at the last newline/space inside the window.
        if end < n:
            window = text[start:end]
            cut = max(window.rfind("\n"), window.rfind(". "))
            if cut > size * 0.5:
                end = start + cut + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = end - overlap
    return chunks


def build_chunks(pages: list[dict]) -> list[dict]:
    chunks: list[dict] = []
    for page in pages:
        context = (page.get("context") or "").strip()
        # Keep short PDFs (winner-list tables) whole so all rows land in one chunk.
        is_pdf = page["url"].lower().endswith(".pdf")
        size = config.PDF_CHUNK_SIZE if is_pdf else config.CHUNK_SIZE
        for piece in chunk_text(page["text"], size, config.CHUNK_OVERLAP):
            # Prepend the doc context to every chunk so bare tables (winner lists)
            # and later chunks keep a searchable anchor, not just the first chunk.
            text = f"{context}\n{piece}" if context else piece
            chunks.append(
                {
                    "id": f"chunk-{len(chunks)}",
                    "url": page["url"],
                    "text": text,
                }
            )
    return chunks


def save_chunks(chunks: list[dict]) -> None:
    with open(config.CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def build_index(chunks: list[dict]) -> None:
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    # Rebuild from scratch each run so re-scrapes don't leave stale chunks.
    try:
        client.delete_collection(config.CHROMA_COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(
        config.CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"}
    )

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks via '{config.EMBEDDING_MODEL}' ...")
    vectors = embed(texts)

    collection.add(
        ids=[c["id"] for c in chunks],
        documents=texts,
        embeddings=vectors,
        metadatas=[{"url": c["url"]} for c in chunks],
    )
    print(f"Indexed {collection.count()} chunks into {config.CHROMA_DIR}")


def main() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    pages = load_pages()
    print(f"Loaded {len(pages)} pages")
    chunks = build_chunks(pages)
    save_chunks(chunks)
    print(f"Created {len(chunks)} chunks -> {config.CHUNKS_PATH}")
    build_index(chunks)
    print("Done. Tip: run a few queries and adjust MAX_DISTANCE in core/config.py.")


if __name__ == "__main__":
    main()
