import logging
import os
import pickle
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from rag.config import settings
from rag.ingestion import Chunk, ParentChunk, load_and_chunk_documents

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL: SentenceTransformer | None = None


class _LocalEmbeddings(Embeddings):
    def __init__(self, model: SentenceTransformer) -> None:
        self._model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(text, show_progress_bar=False).tolist()


def _get_embeddings() -> _LocalEmbeddings:
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _LocalEmbeddings(_EMBEDDING_MODEL)


@dataclass
class HybridIndex:
    faiss: FAISS
    bm25: BM25Okapi
    chunks: list[Chunk]
    parent_chunks: list[ParentChunk]
    chunk_texts: list[str]


def chunk_to_document(c: Chunk) -> Document:
    return Document(
        page_content=c.text,
        metadata={
            "chunk_id": c.id,
            "parent_id": c.parent_id,
            "page_num": c.page_num,
            "source_file": c.source_file,
        },
    )


def build_index() -> HybridIndex:
    chunks, parent_chunks = load_and_chunk_documents()
    chunk_texts = [c.text for c in chunks]

    logger.info("Building FAISS index with %d chunks...", len(chunks))
    embeddings = _get_embeddings()
    faiss_index = FAISS.from_documents(
        [chunk_to_document(c) for c in chunks],
        embeddings,
    )

    logger.info("Building BM25 index...")
    tokenized = [text.split() for text in chunk_texts]
    bm25_index = BM25Okapi(tokenized)

    os.makedirs(settings.VECTOR_STORE_PATH, exist_ok=True)

    faiss_index.save_local(settings.VECTOR_STORE_PATH)
    logger.info("FAISS index saved to %s", settings.VECTOR_STORE_PATH)

    bm25_path = os.path.join(settings.VECTOR_STORE_PATH, "bm25.pkl")
    with open(bm25_path, "wb") as f:
        pickle.dump({
            "bm25": bm25_index,
            "chunks": chunks,
            "parent_chunks": parent_chunks,
            "chunk_texts": chunk_texts,
        }, f)
    logger.info("BM25 index saved to %s", bm25_path)

    return HybridIndex(
        faiss=faiss_index,
        bm25=bm25_index,
        chunks=chunks,
        parent_chunks=parent_chunks,
        chunk_texts=chunk_texts,
    )


def load_index() -> HybridIndex:
    embeddings = _get_embeddings()
    faiss_index = FAISS.load_local(
        settings.VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    logger.info("FAISS index loaded from %s", settings.VECTOR_STORE_PATH)

    bm25_path = os.path.join(settings.VECTOR_STORE_PATH, "bm25.pkl")
    with open(bm25_path, "rb") as f:
        data = pickle.load(f)

    logger.info("BM25 index loaded from %s", bm25_path)
    return HybridIndex(
        faiss=faiss_index,
        bm25=data["bm25"],
        chunks=data["chunks"],
        parent_chunks=data["parent_chunks"],
        chunk_texts=data["chunk_texts"],
    )


def index_exists() -> bool:
    faiss_path = os.path.join(settings.VECTOR_STORE_PATH, "index.faiss")
    bm25_path = os.path.join(settings.VECTOR_STORE_PATH, "bm25.pkl")
    return os.path.exists(faiss_path) and os.path.exists(bm25_path)
