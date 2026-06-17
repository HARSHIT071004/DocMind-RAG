import json
import logging
import time
from typing import Any

from openai import APIStatusError, OpenAI

from rag.cache import query_cache
from rag.config import settings
from rag.hybrid_index import HybridIndex
from rag.memory import format_history_for_prompt, get_recent_history, save_message
from rag.retriever import RetrievalResult, retrieve_hybrid

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """You are a precise document analysis assistant. Answer based ONLY on the provided context.

Rules:
1. Answer concisely and factually. If the context does not contain enough information, say "I cannot find this in the documents."
2. Cite your sources inline using [source: filename, page N] for each claim.
3. Respond in JSON format only:
{
  "answer": "your concise answer here",
  "citations": [{"source": "filename.pdf", "page": 5}],
  "confidence": 0.95
}
4. confidence must be a float 0.0-1.0 based on how well the context supports your answer.
5. Do NOT add any text outside the JSON block."""  # noqa: E501


def _build_llm_client() -> OpenAI:
    return OpenAI(
        api_key=settings.OPENROUTER_API_KEY,
        base_url=settings.LLM_BASE_URL,
        timeout=settings.LLM_TIMEOUT_SECONDS,
    )


def _format_context(results: list[RetrievalResult]) -> str:
    blocks = []
    for i, r in enumerate(results, 1):
        blocks.append(
            f"[{i}] Source: {r.source_file}, page {r.page_num}\n{r.parent_text}"
        )
    return "\n\n".join(blocks)


def _compute_confidence(
    results: list[RetrievalResult],
    llm_confidence: float = 0.5,
) -> float:
    if not results:
        return 0.0

    retrieval_scores = [r.rrf_score for r in results if r.rrf_score > 0]
    avg_retrieval = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0.0

    normalized_retrieval = min(1.0, avg_retrieval * 50)
    confidence = 0.4 * normalized_retrieval + 0.6 * llm_confidence
    return round(min(1.0, confidence), 2)


def _call_llm(client: OpenAI, model: str, prompt: str) -> str | None:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=settings.MAX_TOKENS,


        )
        return resp.choices[0].message.content
    except APIStatusError as e:
        if e.status_code == 429:
            logger.warning("Rate-limited on %s", model)
        else:
            logger.error("LLM error on %s: %s", model, e)
        return None
    except Exception as e:
        logger.error("LLM error on %s: %s", model, e)
        return None


def _parse_response(raw: str | None) -> dict:
    if not raw:
        return {"answer": "I could not generate an answer at this time.", "citations": [], "confidence": 0.0}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"answer": raw, "citations": [], "confidence": 0.0}


def answer(
    question: str,
    index: HybridIndex,
    session_id: str | None = None,
) -> dict[str, Any]:
    cached = query_cache.get(question)
    if cached:
        logger.info("Cache hit for question: %s", question[:60])
        return cached

    context_results = retrieve_hybrid(question, index)
    context_text = _format_context(context_results)

    history_text = ""
    if session_id:
        history = get_recent_history(session_id, settings.MEMORY_MAX_TURNS)
        if history:
            history_text = format_history_for_prompt(history, settings.MEMORY_MAX_TOKENS)

    user_prompt = ""
    if history_text:
        user_prompt += f"Previous conversation:\n{history_text}\n\n"
    user_prompt += f"Context:\n{context_text}\n\nQuestion: {question}"

    client = _build_llm_client()
    models = [settings.LLM_MODEL] + settings.LLM_FALLBACK_MODELS
    raw_response = None

    for attempt, model in enumerate(models):
        raw_response = _call_llm(client, model, user_prompt)
        if raw_response:
            break
        if attempt < len(models) - 1:
            wait = 10 + (attempt * 5)
            logger.info("Waiting %ds before next model...", wait)
            time.sleep(wait)

    parsed = _parse_response(raw_response)
    llm_conf = float(parsed.get("confidence", 0.5))
    parsed["confidence"] = _compute_confidence(context_results, llm_conf)
    parsed["retrieved_chunks"] = [
        {
            "chunk_id": r.chunk_id,
            "source": r.source_file,
            "page": r.page_num,
            "text": r.chunk_text[:200],
        }
        for r in context_results
    ]

    if (
        settings.CRAG_ENABLED
        and parsed["confidence"] < settings.CONFIDENCE_THRESHOLD
    ):
        logger.info("Confidence %.2f below threshold. Running CRAG re-retrieval...", parsed["confidence"])
        rephrased = _rephrase_query(question, client)
        if rephrased and rephrased != question:
            new_results = retrieve_hybrid(rephrased, index)
            new_context = _format_context(new_results)
            new_prompt = f"Context:\n{new_context}\n\nQuestion: {question}"
            new_raw = _call_llm(client, settings.CRAG_RETRY_MODEL, new_prompt)
            new_parsed = _parse_response(new_raw)
            if new_parsed.get("answer"):
                new_llm_conf = float(new_parsed.get("confidence", 0.5))
                parsed["answer"] = new_parsed["answer"]
                parsed["citations"] = new_parsed.get("citations", [])
                parsed["confidence"] = _compute_confidence(new_results, new_llm_conf)
                parsed["retrieved_chunks"] = [
                    {
                        "chunk_id": r.chunk_id,
                        "source": r.source_file,
                        "page": r.page_num,
                        "text": r.chunk_text[:200],
                    }
                    for r in new_results
                ]
                parsed["crag_retried"] = True
                logger.info("CRAG improved confidence to %.2f", parsed["confidence"])

    if session_id:
        save_message(session_id, "user", question)
        save_message(session_id, "assistant", json.dumps({
            "answer": parsed.get("answer", ""),
            "citations": parsed.get("citations", []),
        }))
        if not history_text:
            session_title = question[:50] + ("..." if len(question) > 50 else "")
            from rag.memory import _get_conn
            conn = _get_conn()
            conn.execute("UPDATE sessions SET title=? WHERE id=?", (session_title, session_id))
            conn.commit()
            conn.close()

    query_cache.set(question, parsed)
    return parsed


def _rephrase_query(question: str, client: OpenAI) -> str:
    try:
        resp = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": "Rephrase the following question to improve document retrieval. Return ONLY the rephrased question, no explanation."},
                {"role": "user", "content": question},
            ],
            temperature=0.3,
            max_tokens=100,
        )
        rephrased = resp.choices[0].message.content.strip()
        if rephrased and rephrased != question:
            logger.info("CRAG rephrased: '%s' -> '%s'", question[:50], rephrased[:50])
            return rephrased
    except Exception as e:
        logger.warning("CRAG rephrase failed: %s", e)
    return question
