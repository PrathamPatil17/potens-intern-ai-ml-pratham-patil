"""ChromaDB persistence and query wrapper."""
from dataclasses import dataclass
import chromadb
from .config import CHROMA_DIR, COLLECTION_NAME
from .chunker import Chunk

# Cosine space so distances line up with normalized embeddings and
# config.DISTANCE_THRESHOLD.
_METADATA = {"hnsw:space": "cosine"}


@dataclass
class Retrieved:
    doc_id: str
    source_file: str
    section_title: str
    chunk_index: int
    text: str
    distance: float


def _client() -> chromadb.ClientAPI:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection():
    return _client().get_or_create_collection(COLLECTION_NAME, metadata=_METADATA)


def reset_collection():
    """Delete and recreate the collection so ingest is idempotent."""
    client = _client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # Collection may not exist yet on first run.
    return client.get_or_create_collection(COLLECTION_NAME, metadata=_METADATA)


def add_chunks(chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    col = get_collection()
    # Chroma ids must be unique strings; combine doc_id and chunk_index.
    ids = [f"{c.doc_id}:{c.chunk_index}" for c in chunks]
    metadatas = [{
        "doc_id": c.doc_id,
        "source_file": c.source_file,
        "section_title": c.section_title,
        "chunk_index": c.chunk_index,
    } for c in chunks]
    documents = [c.text for c in chunks]
    col.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)


def query(query_embedding: list[float], top_k: int,
          doc_id: str | None = None) -> list[Retrieved]:
    col = get_collection()
    where = {"doc_id": doc_id} if doc_id else None
    res = col.query(query_embeddings=[query_embedding], n_results=top_k, where=where)
    out: list[Retrieved] = []
    # Chroma returns parallel lists nested one level per query.
    metas = res["metadatas"][0]
    docs = res["documents"][0]
    dists = res["distances"][0]
    for meta, text, dist in zip(metas, docs, dists):
        out.append(Retrieved(
            doc_id=meta["doc_id"],
            source_file=meta["source_file"],
            section_title=meta["section_title"],
            chunk_index=meta["chunk_index"],
            text=text,
            distance=dist,
        ))
    return out


def list_doc_ids() -> list[str]:
    col = get_collection()
    got = col.get(include=["metadatas"])
    return sorted({m["doc_id"] for m in got["metadatas"]})
