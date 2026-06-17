from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class EvalSample:
    question: str
    ground_truth: str
    answer: str = ""
    contexts: list[str] = field(default_factory=list)
    confidence: float = 0.0
    latency_ms: float = 0.0
    error: str | None = None


@dataclass
class EvalResult:
    samples: list[EvalSample]
    metrics: dict[str, float] = field(default_factory=dict)
    timestamp: str = ""


def load_test_set(path: str | None = None) -> list[dict]:
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return [
        {"question": "What is the main topic of the first document?", "ground_truth": ""},
        {"question": "What technical skills are mentioned?", "ground_truth": ""},
        {"question": "What experience does the candidate have?", "ground_truth": ""},
    ]


def run_evaluation(
    test_set: list[dict],
    index: Any,
    output_path: str = "evaluation_results.json",
) -> EvalResult:
    from rag import answer

    samples: list[EvalSample] = []
    total = len(test_set)

    for i, item in enumerate(test_set):
        question = item["question"]
        ground_truth = item.get("ground_truth", "")
        logger.info("Evaluating [%d/%d]: %s", i + 1, total, question[:60])

        sample = EvalSample(question=question, ground_truth=ground_truth)
        start = time.monotonic()

        try:
            result = answer(question, index)
            sample.answer = result.get("answer", "")
            sample.confidence = result.get("confidence", 0.0)
            raw_chunks = result.get("retrieved_chunks", [])
            sample.contexts = [c.get("text", "") for c in raw_chunks if c.get("text")]
        except Exception as e:
            logger.error("Eval failed for '%s': %s", question[:40], e)
            sample.error = str(e)

        sample.latency_ms = round((time.monotonic() - start) * 1000, 1)
        samples.append(sample)

    result = _compute_ragas_metrics(samples)

    result.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    _save_results(result, output_path)
    return result


def _compute_ragas_metrics(samples: list[EvalSample]) -> EvalResult:
    completed = [s for s in samples if s.answer and not s.error]
    if not completed:
        logger.warning("No successful samples to evaluate")
        return EvalResult(samples=samples, metrics={"error_rate": 1.0})

    try:
        from datasets import Dataset
        from langchain_openai import ChatOpenAI
        from ragas import evaluate
        from ragas.llms.base import LangchainLLMWrapper
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        from rag.config import settings

        llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            openai_api_key=settings.OPENROUTER_API_KEY,
            openai_api_base=settings.LLM_BASE_URL,
            temperature=0,
        )
        ragas_llm = LangchainLLMWrapper(llm)

        for metric in [faithfulness, answer_relevancy, context_precision, context_recall]:
            if hasattr(metric, "llm"):
                metric.llm = ragas_llm

        data = {
            "question": [s.question for s in completed],
            "answer": [s.answer for s in completed],
            "contexts": [s.contexts for s in completed],
            "ground_truth": [s.ground_truth or s.answer for s in completed],
        }
        dataset = Dataset.from_dict(data)
        logger.info("Running RAGAS metrics on %d samples...", len(completed))

        ragas_result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=ragas_llm,
            raise_exceptions=False,
        )

        metrics_map = {
            "faithfulness": faithfulness.name,
            "answer_relevancy": answer_relevancy.name,
            "context_precision": context_precision.name,
            "context_recall": context_recall.name,
        }
        metrics = {}
        for key, metric_name in metrics_map.items():
            try:
                scores = ragas_result[metric_name]
                metrics[key] = _safe_mean(scores)
            except Exception:
                metrics[key] = 0.0

        metrics["error_rate"] = round(1 - len(completed) / len(samples), 3)
        metrics["num_samples"] = len(completed)
        metrics["total_questions"] = len(samples)

        logger.info("RAGAS results: %s", metrics)

    except Exception as e:
        logger.error("RAGAS computation failed: %s", e)
        metrics = {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "error_rate": round(1 - len(completed) / len(samples), 3),
            "num_samples": len(completed),
            "total_questions": len(samples),
            "ragas_error": str(e),
        }

    return EvalResult(samples=samples, metrics=metrics)


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    import math
    filtered = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not filtered:
        return 0.0
    return round(sum(filtered) / len(filtered), 4)


def _save_results(result: EvalResult, path: str) -> None:
    output = {
        "timestamp": result.timestamp,
        "metrics": result.metrics,
        "samples": [
            {
                "question": s.question,
                "ground_truth": s.ground_truth,
                "answer": s.answer[:300] if s.answer else "",
                "confidence": s.confidence,
                "latency_ms": s.latency_ms,
                "error": s.error,
            }
            for s in result.samples
        ],
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", path)


if __name__ == "__main__":
    test_path = os.environ.get("EVAL_TEST_SET")
    test_set = load_test_set(test_path)

    from rag import load_index, index_exists

    if not index_exists():
        logger.error("No index found. Build the index first.")
        sys.exit(1)

    index = load_index()
    result = run_evaluation(test_set, index)

    print("\n=== EVALUATION RESULTS ===")
    for name, val in result.metrics.items():
        print(f"  {name}: {val}")
    print(f"\nResults saved to evaluation_results.json")
