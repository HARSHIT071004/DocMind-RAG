from rag.hybrid_index import build_index, index_exists, load_index, HybridIndex
from rag.retriever import retrieve_faiss, retrieve_bm25, retrieve_hybrid, RetrievalResult
from rag.pipeline import answer

build_vector_store = build_index
is_vector_store_ready = index_exists
load_vector_store = load_index

__all__ = [
    "build_index",
    "build_vector_store",
    "index_exists",
    "is_vector_store_ready",
    "load_index",
    "load_vector_store",
    "HybridIndex",
    "retrieve_faiss",
    "retrieve_bm25",
    "retrieve_hybrid",
    "RetrievalResult",
    "answer",
]
