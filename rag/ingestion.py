# rag/ingestion.py — PDF ingestion pipeline
# Responsibilities: load PDFs → chunk → embed → build FAISS → persist to disk

import logging
import os

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import settings

logger = logging.getLogger(__name__)


def _get_embeddings() -> OpenAIEmbeddings:
    """Initialise the embedding model via OpenRouter."""
    return OpenAIEmbeddings(
        api_key=settings.OPENROUTER_API_KEY,
        model=settings.EMBEDDING_MODEL,
        base_url="https://openrouter.ai/api/v1",
    )


def build_vector_store() -> FAISS:
    """
    Full ingestion pipeline:
      1. Load PDFs from ARTIFACTS_DIR
      2. Split into chunks
      3. Embed with Google Generative AI
      4. Build FAISS index and persist to VECTOR_STORE_PATH
    Returns the FAISS vector store.
    """
    logger.info("Loading PDFs from %s", settings.ARTIFACTS_DIR)
    loader = PyPDFDirectoryLoader(settings.ARTIFACTS_DIR)
    docs = loader.load()

    if not docs:
        raise ValueError(f"No PDF documents found in '{settings.ARTIFACTS_DIR}'")

    # Chunk documents (cap at MAX_PAGES pages)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs[: settings.MAX_PAGES])
    logger.info("Created %d chunks from %d pages", len(chunks), min(len(docs), settings.MAX_PAGES))

    # Build FAISS index
    embeddings = _get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Persist so it survives app restarts
    vector_store.save_local(settings.VECTOR_STORE_PATH)
    logger.info("Vector store saved to %s", settings.VECTOR_STORE_PATH)

    return vector_store


def load_vector_store() -> FAISS:
    """
    Load a previously persisted FAISS index from disk.
    Raises FileNotFoundError if the index does not exist yet.
    """
    index_file = os.path.join(settings.VECTOR_STORE_PATH, "index.faiss")
    if not os.path.exists(index_file):
        raise FileNotFoundError(
            f"No vector store found at '{settings.VECTOR_STORE_PATH}'. "
            "Run ingestion first."
        )
    embeddings = _get_embeddings()
    vector_store = FAISS.load_local(
        settings.VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    logger.info("Vector store loaded from %s", settings.VECTOR_STORE_PATH)
    return vector_store


def is_vector_store_ready() -> bool:
    """Return True if a persisted FAISS index already exists on disk."""
    return os.path.exists(os.path.join(settings.VECTOR_STORE_PATH, "index.faiss"))
