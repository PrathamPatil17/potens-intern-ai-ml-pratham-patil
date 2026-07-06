"""Local sentence-transformers embeddings. Model is loaded lazily and reused."""
from sentence_transformers import SentenceTransformer
from .config import EMBED_MODEL

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    return _get_model().encode(texts, normalize_embeddings=True).tolist()


def embed_query(text: str) -> list[float]:
    return _get_model().encode([text], normalize_embeddings=True)[0].tolist()
