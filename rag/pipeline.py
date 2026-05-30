# rag/pipeline.py — retrieval pipeline
# Responsibilities: initialise LLM, build retrieval chain, run Q&A

import logging
import time

from openai import APIStatusError
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS

from rag.config import settings

logger = logging.getLogger(__name__)

# ── Prompt ────────────────────────────────────────────────────────
_PROMPT_TEMPLATE = """
Answer concisely based ONLY on the provided context.
If the context doesn't contain the answer, say "I cannot find this in the documents."
<context>
{context}
</context>
Question: {input}
"""

_prompt = ChatPromptTemplate.from_template(_PROMPT_TEMPLATE)


def _build_llm(model: str | None = None) -> ChatOpenAI:
    """Initialise LLM via OpenRouter (OpenAI-compatible endpoint)."""
    return ChatOpenAI(
        api_key=settings.OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        model=model or settings.LLM_MODEL,
        max_tokens=settings.MAX_TOKENS,
        max_retries=0,
    )


def _is_rate_limit_error(e: Exception) -> bool:
    """Check if the exception is a 429 rate-limit from OpenRouter."""
    if isinstance(e, APIStatusError) and e.status_code == 429:
        return True
    return False


def answer(question: str, vector_store: FAISS) -> dict:
    """
    Run the full RAG retrieval pipeline with retry & fallback models.

    Tries models in order: LLM_MODEL → each LLM_FALLBACK_MODELS.
    On 429 (rate-limit) waits with exponential backoff then tries next model.

    Returns the raw chain response dict with keys:
      - 'answer'  : the generated answer string
      - 'context' : list of retrieved Document chunks (for citations)
      - 'input'   : the original question
    """
    models_to_try = [settings.LLM_MODEL, *settings.LLM_FALLBACK_MODELS]
    last_error = None

    for attempt, model in enumerate(models_to_try):
        try:
            logger.info(
                "Attempt %d/%d — trying model: %s",
                attempt + 1, len(models_to_try), model,
            )

            llm = _build_llm(model)
            document_chain = create_stuff_documents_chain(llm, _prompt)
            retriever = vector_store.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            logger.info("Invoking chain for question: %s", question)
            response = retrieval_chain.invoke({"input": question})
            logger.info(
                "Chain OK — model=%s answer_len=%d",
                model, len(response.get("answer", "")),
            )
            return response

        except Exception as e:
            last_error = e
            logger.warning("Model %s failed: %s", model, e)

            if _is_rate_limit_error(e):
                if attempt < len(models_to_try) - 1:
                    wait = 10 + (attempt * 5)
                    logger.info(
                        "Rate-limited on %s — waiting %ds then trying next model",
                        model, wait,
                    )
                    time.sleep(wait)
                    continue
            else:
                # Non-rate-limit error — don't retry with other models
                break

    # All attempts exhausted
    logger.error("All %d models failed. Last error: %s", len(models_to_try), last_error)
    raise last_error or RuntimeError("All models failed")
