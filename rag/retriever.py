import logging
import math
from dataclasses import dataclass

from rag.config import settings
from rag.hybrid_index import HybridIndex

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    chunk_id: int
    chunk_text: str
    parent_id: int
    parent_text: str
    page_num: int
    source_file: str
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rrf_score: float = 0.0


def retrieve_faiss(query: str, index: HybridIndex, k: int | None = None) -> list[RetrievalResult]:
    k = k or settings.TOP_K_FAISS
    docs_and_scores = index.faiss.similarity_search_with_score(query, k=k)

    results = []
    for doc, score in docs_and_scores:
        meta = doc.metadata
        chunk_id = meta["chunk_id"]
        matching = [c for c in index.chunks if c.id == chunk_id]
        if not matching:
            continue
        chunk = matching[0]
        parent = next(
            (p for p in index.parent_chunks if p.id == chunk.parent_id),
            None,
        )
        results.append(RetrievalResult(
            chunk_id=chunk.id,
            chunk_text=chunk.text,
            parent_id=chunk.parent_id,
            parent_text=parent.text if parent else "",
            page_num=chunk.page_num,
            source_file=chunk.source_file,
            dense_score=float(score),
        ))

    return results


def retrieve_bm25(query: str, index: HybridIndex, k: int | None = None) -> list[RetrievalResult]:
    k = k or settings.TOP_K_BM25
    tokenized_query = query.split()
    scores = index.bm25.get_scores(tokenized_query)

    indexed = list(enumerate(scores))
    indexed.sort(key=lambda x: x[1], reverse=True)

    results = []
    for idx, score in indexed[:k]:
        chunk = index.chunks[idx]
        parent = next(
            (p for p in index.parent_chunks if p.id == chunk.parent_id),
            None,
        )
        results.append(RetrievalResult(
            chunk_id=chunk.id,
            chunk_text=chunk.text,
            parent_id=chunk.parent_id,
            parent_text=parent.text if parent else "",
            page_num=chunk.page_num,
            source_file=chunk.source_file,
            sparse_score=float(score),
        ))

    return results


def _rrf_merge(
    dense_results: list[RetrievalResult],
    sparse_results: list[RetrievalResult],
    k: int = 60,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    top_n: int = 12,
) -> list[RetrievalResult]:
    dense_rank = {r.chunk_id: i for i, r in enumerate(dense_results)}
    sparse_rank = {r.chunk_id: i for i, r in enumerate(sparse_results)}
    all_ids = set(dense_rank.keys()) | set(sparse_rank.keys())

    scored = []
    for cid in all_ids:
        dr = dense_rank.get(cid, 1_000_000)
        sr = sparse_rank.get(cid, 1_000_000)
        rrf = dense_weight / (k + dr) + sparse_weight / (k + sr)
        scored.append((cid, rrf))

    scored.sort(key=lambda x: x[1], reverse=True)
    top_ids = {cid for cid, _ in scored[:top_n]}

    merged = {}
    for r in dense_results:
        if r.chunk_id in top_ids:
            merged[r.chunk_id] = r
            merged[r.chunk_id].rrf_score = 0.0
    for r in sparse_results:
        if r.chunk_id in top_ids:
            if r.chunk_id in merged:
                merged[r.chunk_id].sparse_score = r.sparse_score
            else:
                merged[r.chunk_id] = r
                merged[r.chunk_id].rrf_score = 0.0

    for cid, rrf in scored[:top_n]:
        if cid in merged:
            merged[cid].rrf_score = rrf

    result = sorted(merged.values(), key=lambda x: x.rrf_score, reverse=True)
    return result


def _mmr_diversify(
    results: list[RetrievalResult],
    query_embedding: list[float],
    lambda_mult: float = 0.7,
    top_n: int = 6,
) -> list[RetrievalResult]:
    if len(results) <= top_n:
        return results[:top_n]

    import numpy as np
    q_vec = np.array(query_embedding, dtype="float32")
    chunk_vecs = {}
    from rag.hybrid_index import _get_embeddings as _get_emb
    emb = _get_emb()
    for r in results:
        v = np.array(emb.embed_query(r.chunk_text), dtype="float32")
        chunk_vecs[r.chunk_id] = v / (np.linalg.norm(v) + 1e-12)

    selected = []
    remaining = list(results)

    while len(selected) < top_n and remaining:
        best_score = -1e9
        best_idx = -1
        for i, r in enumerate(remaining):
            sim_q = float(np.dot(chunk_vecs[r.chunk_id], q_vec))
            if selected:
                sim_s = max(
                    float(np.dot(chunk_vecs[r.chunk_id], chunk_vecs[s.chunk_id]))
                    for s in selected
                )
            else:
                sim_s = 0.0
            mmr = lambda_mult * sim_q - (1 - lambda_mult) * sim_s
            if mmr > best_score:
                best_score = mmr
                best_idx = i
        selected.append(remaining.pop(best_idx))

    return selected


_CROSS_ENCODER = None


def _get_cross_encoder():
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        try:
            from sentence_transformers import CrossEncoder
            _CROSS_ENCODER = CrossEncoder(
                settings.CROSS_ENCODER_MODEL,
                max_length=512,
            )
            logger.info("Cross-encoder loaded: %s", settings.CROSS_ENCODER_MODEL)
        except Exception as e:
            logger.warning("Failed to load cross-encoder: %s", e)
            return None
    return _CROSS_ENCODER


def _rerank_with_cross_encoder(
    query: str,
    results: list[RetrievalResult],
    top_n: int = 4,
) -> list[RetrievalResult]:
    model = _get_cross_encoder()
    if model is None:
        return results[:top_n]

    pairs = [(query, r.parent_text) for r in results]
    scores = model.predict(pairs, show_progress_bar=False)

    for r, s in zip(results, scores):
        r.rrf_score = float(s)

    results.sort(key=lambda x: x.rrf_score, reverse=True)
    return results[:top_n]


def retrieve_hybrid(
    query: str,
    index: HybridIndex,
    enable_mmr: bool = True,
    enable_cross_encoder: bool = True,
) -> list[RetrievalResult]:
    dense_results = retrieve_faiss(query, index, k=settings.TOP_K_FAISS)
    sparse_results = retrieve_bm25(query, index, k=settings.TOP_K_BM25)

    rrf_results = _rrf_merge(
        dense_results,
        sparse_results,
        k=settings.RRF_K,
        dense_weight=settings.RRF_DENSE_WEIGHT,
        sparse_weight=settings.RRF_SPARSE_WEIGHT,
        top_n=settings.MMR_NUM_DIVERSIFY if enable_mmr else settings.TOP_K_HYBRID,
    )

    if enable_mmr and rrf_results:
        dense_query_vec = index.faiss.embedding_function.embed_query(query)
        rrf_results = _mmr_diversify(
            rrf_results,
            dense_query_vec,
            lambda_mult=settings.MMR_LAMBDA,
            top_n=settings.MMR_NUM_DIVERSIFY,
        )

    if enable_cross_encoder and settings.ENABLE_CROSS_ENCODER:
        rrf_results = _rerank_with_cross_encoder(query, rrf_results, settings.TOP_K_HYBRID)
    else:
        rrf_results = rrf_results[:settings.TOP_K_HYBRID]

    logger.info(
        "Hybrid: FAISS=%d BM25=%d -> RRF=%d -> final=%d",
        len(dense_results), len(sparse_results), len(rrf_results), len(rrf_results),
    )
    return rrf_results
