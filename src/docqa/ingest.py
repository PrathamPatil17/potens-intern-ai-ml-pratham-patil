"""One-time ingestion: load each source document, chunk it, embed the chunks,
and store them in a fresh Chroma collection. Idempotent: rebuilds from scratch.

Run:  python -m src.docqa.ingest
"""
from .config import DOCUMENTS, DOCUMENTS_DIR
from .chunker import chunk_document
from .embeddings import embed_texts
from . import store


def run() -> int:
    store.reset_collection()
    total = 0
    for doc in DOCUMENTS:
        path = DOCUMENTS_DIR / doc["source_file"]
        text = path.read_text(encoding="utf-8")
        chunks = chunk_document(text, doc["doc_id"], doc["source_file"])
        vectors = embed_texts([c.text for c in chunks])
        store.add_chunks(chunks, vectors)
        print(f"  {doc['doc_id']}: {len(chunks)} chunks")
        total += len(chunks)
    print(f"Ingested {total} chunks from {len(DOCUMENTS)} documents.")
    return total


if __name__ == "__main__":
    run()
