from __future__ import annotations

import logging
import json
import re
import statistics
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .rag_pipeline import FoodRAGPipeline
    from .rag_pipeline import OllamaClient

logger = logging.getLogger(__name__)


@dataclass
class EvalSample:
    query: str
    ground_truth_answer: str
    relevant_doc_ids: list[str]
    retrieved_doc_ids: list[str]
    generated_answer: str
    retrieved_contexts: list[str]


@dataclass
class RetrievalMetrics:
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    mrr: float = 0.0
    hit_rate: float = 0.0
    k: int = 5


@dataclass
class GenerationMetrics:
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    answer_correctness: float = 0.0


@dataclass
class EvalReport:
    n_samples: int = 0
    retrieval: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    generation: GenerationMetrics = field(default_factory=GenerationMetrics)
    passed_targets: list[str] = field(default_factory=list)
    failed_targets: list[str] = field(default_factory=list)

    TARGETS: dict[str, float] = field(default_factory=lambda: {
        "recall_at_k": 0.80,
        "precision_at_k": 0.60,
        "mrr": 0.70,
        "hit_rate": 0.90,
        "faithfulness": 0.70,
        "answer_relevancy": 0.60,
        "context_precision": 0.60,
        "context_recall": 0.60,
        "answer_correctness": 0.50,
    })

    def compute_pass_fail(self) -> None:
        metrics = {
            "recall_at_k": self.retrieval.recall_at_k,
            "precision_at_k": self.retrieval.precision_at_k,
            "mrr": self.retrieval.mrr,
            "hit_rate": self.retrieval.hit_rate,
            "faithfulness": self.generation.faithfulness,
            "answer_relevancy": self.generation.answer_relevancy,
            "context_precision": self.generation.context_precision,
            "context_recall": self.generation.context_recall,
            "answer_correctness": self.generation.answer_correctness,
        }
        self.passed_targets = [k for k, v in metrics.items() if v >= self.TARGETS[k]]
        self.failed_targets = [k for k, v in metrics.items() if v < self.TARGETS[k]]

    def summary(self) -> str:
        lines = [
            f"=== RAG Eval Report ({self.n_samples} samples) ===",
            "",
            "[ Retrieval ]",
            f"  Recall@{self.retrieval.k}:    {self.retrieval.recall_at_k:.3f}  (target >= {self.TARGETS['recall_at_k']})",
            f"  Precision@{self.retrieval.k}: {self.retrieval.precision_at_k:.3f}  (target >= {self.TARGETS['precision_at_k']})",
            f"  MRR:         {self.retrieval.mrr:.3f}  (target >= {self.TARGETS['mrr']})",
            f"  Hit Rate:    {self.retrieval.hit_rate:.3f}  (target >= {self.TARGETS['hit_rate']})",
            "",
            "[ Generation ]",
            f"  Faithfulness:      {self.generation.faithfulness:.3f}  (target >= {self.TARGETS['faithfulness']})",
            f"  Answer Relevancy:  {self.generation.answer_relevancy:.3f}  (target >= {self.TARGETS['answer_relevancy']})",
            f"  Context Precision: {self.generation.context_precision:.3f}  (target >= {self.TARGETS['context_precision']})",
            f"  Context Recall:    {self.generation.context_recall:.3f}  (target >= {self.TARGETS['context_recall']})",
            f"  Answer Correctness:{self.generation.answer_correctness:.3f}  (target >= {self.TARGETS['answer_correctness']})",
            "",
            f"PASSED: {', '.join(self.passed_targets) or 'none'}",
            f"FAILED: {', '.join(self.failed_targets) or 'none'}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "retrieval": {
                "recall_at_k": self.retrieval.recall_at_k,
                "precision_at_k": self.retrieval.precision_at_k,
                "mrr": self.retrieval.mrr,
                "hit_rate": self.retrieval.hit_rate,
                "k": self.retrieval.k,
            },
            "generation": {
                "faithfulness": self.generation.faithfulness,
                "answer_relevancy": self.generation.answer_relevancy,
                "context_precision": self.generation.context_precision,
                "context_recall": self.generation.context_recall,
                "answer_correctness": self.generation.answer_correctness,
            },
            "passed_targets": self.passed_targets,
            "failed_targets": self.failed_targets,
        }


def evaluate_retrieval(samples: list[EvalSample], k: int = 5) -> RetrievalMetrics:
    if not samples:
        return RetrievalMetrics(k=k)

    recalls, precisions, reciprocal_ranks, hits = [], [], [], []

    for sample in samples:
        relevant = set(sample.relevant_doc_ids)
        retrieved = sample.retrieved_doc_ids[:k]

        if not relevant:
            continue

        hits_in_top_k = [r for r in retrieved if r in relevant]

        recalls.append(len(hits_in_top_k) / len(relevant))
        judged_k = min(k, len(retrieved), len(relevant))
        precisions.append(len(hits_in_top_k) / judged_k if judged_k > 0 else 0.0)
        hits.append(1.0 if hits_in_top_k else 0.0)

        rr = 0.0
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    def safe_mean(lst: list[float]) -> float:
        return statistics.mean(lst) if lst else 0.0

    return RetrievalMetrics(
        recall_at_k=safe_mean(recalls),
        precision_at_k=safe_mean(precisions),
        mrr=safe_mean(reciprocal_ranks),
        hit_rate=safe_mean(hits),
        k=k,
    )


def run_full_eval(
    samples: list[EvalSample],
    llm_client: "OllamaClient | None" = None,
    k: int = 5,
) -> EvalReport:
    retrieval = evaluate_retrieval(samples, k=k)
    generation = evaluate_generation(samples, llm_client=llm_client)
    report = EvalReport(
        n_samples=len(samples),
        retrieval=retrieval,
        generation=generation,
    )
    report.compute_pass_fail()
    return report


def evaluate_generation(
    samples: list[EvalSample],
    llm_client: "OllamaClient | None" = None,
) -> GenerationMetrics:
    if not samples:
        return GenerationMetrics()

    faithfulness_scores: list[float] = []
    relevancy_scores: list[float] = []
    ctx_precision_scores: list[float] = []
    ctx_recall_scores: list[float] = []
    correctness_scores: list[float] = []

    for sample in samples:
        judged = _llm_judge_scores(llm_client, sample) if llm_client is not None else None
        if judged is None:
            judged = {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "answer_correctness": 0.0,
            }
        faithfulness_scores.append(judged["faithfulness"])
        relevancy_scores.append(judged["answer_relevancy"])
        ctx_precision_scores.append(judged["context_precision"])
        ctx_recall_scores.append(judged["context_recall"])
        correctness_scores.append(judged["answer_correctness"])

    return GenerationMetrics(
        faithfulness=_mean(faithfulness_scores),
        answer_relevancy=_mean(relevancy_scores),
        context_precision=_mean(ctx_precision_scores),
        context_recall=_mean(ctx_recall_scores),
        answer_correctness=_mean(correctness_scores),
    )


def _llm_judge_scores(
    llm_client: "OllamaClient",
    sample: EvalSample,
) -> dict[str, float] | None:
    try:
        raw = llm_client.generate(
            _judge_prompt(sample),
            json_mode=True,
            max_output_tokens=256,
            temperature=0.0,
        ).strip()
        parsed = _parse_score_json(raw)
        if not isinstance(parsed, dict):
            logger.warning("LLM judge returned non-JSON response: %s", raw[:300])
            return None
        return {
            "faithfulness": _clamp_score(parsed.get("faithfulness")),
            "answer_relevancy": _clamp_score(parsed.get("answer_relevancy")),
            "context_precision": _clamp_score(parsed.get("context_precision")),
            "context_recall": _clamp_score(parsed.get("context_recall")),
            "answer_correctness": _clamp_score(parsed.get("answer_correctness")),
        }
    except Exception as exc:
        logger.warning("LLM judge call failed: %s", exc)
        return None


def _judge_prompt(sample: EvalSample) -> str:
    context = "\n\n".join(sample.retrieved_contexts[:5])
    return f"""Evaluate a food RAG answer. Return one JSON object only.

Schema:
{{
  "faithfulness": 0.0,
  "answer_relevancy": 0.0,
  "context_precision": 0.0,
  "context_recall": 0.0,
  "answer_correctness": 0.0
}}

Use numbers from 0.0 to 1.0.
faithfulness = answer claims are supported by context.
answer_relevancy = answer addresses query.
context_precision = context is relevant to query.
context_recall = context contains ground truth.
answer_correctness = answer matches ground truth.
Penalize invented foods or unsupported nutrition facts.

QUERY: {sample.query}
CONTEXT: {context[:2200]}
GROUND_TRUTH: {sample.ground_truth_answer}
ANSWER: {sample.generated_answer}
JSON:"""


def build_eval_samples(
    pipeline: "FoodRAGPipeline",
    raw_samples: list[Any],
    n_results: int = 5,
) -> list[EvalSample]:
    from .filters import parse_query
    from .query_transformer import QueryRoute, transform_query
    from .rag_pipeline import build_prompt, prepare_context
    from .reranker import normalize_search_result, rerank_results

    eval_samples: list[EvalSample] = []

    for raw_sample in raw_samples:
        query = str(_sample_value(raw_sample, "query", "")).strip()
        if not query:
            logger.warning("Skipping eval sample with empty query")
            continue

        ground_truth = str(_sample_value(raw_sample, "ground_truth_answer", ""))
        raw_relevant_ids = _sample_value(raw_sample, "relevant_doc_ids", []) or []
        relevant_doc_ids = [str(doc_id) for doc_id in raw_relevant_ids]

        filters = parse_query(query)
        transformed = transform_query(query, pipeline.llm_client)
        retrieved_ids: list[str] = []
        contexts: list[str] = []

        if transformed.route in {
            QueryRoute.NORMAL_RETRIEVAL,
            QueryRoute.MULTI_QUERY,
            QueryRoute.MULTI_QUERY_STEPBACK,
        }:
            raw_results = pipeline._retrieve_candidates(transformed, filters, n_results)
            food_results = [normalize_search_result(result) for result in raw_results]
            ranked = rerank_results(
                food_results,
                filters,
                query=query,
                cross_encoder=pipeline.cross_encoder,
            )

            top_ranked = ranked[:n_results]
            retrieved_ids = [item.food_id for item in top_ranked]
            contexts = [_eval_context(item) for item in top_ranked]
            context = prepare_context(top_ranked, top_k=n_results)
            prompt = build_prompt(query, context, filters, transformed)
        elif transformed.route == QueryRoute.CLARIFICATION:
            prompt = (
                "Could you give me a bit more detail? For example: "
                "what cuisine, dietary preference, or calorie range are you looking for?"
            )
        elif transformed.route == QueryRoute.REJECTION:
            prompt = (
                "I'm a food recommendation assistant. "
                "I can help you find dishes, recipes, and nutrition information. "
                "Please ask me something food-related!"
            )

        if transformed.route in {QueryRoute.CLARIFICATION, QueryRoute.REJECTION}:
            answer = prompt
        else:
            try:
                answer = pipeline.llm_client.generate(prompt)
            except Exception as exc:
                answer = f"Generation error: {exc}"

        eval_samples.append(EvalSample(
            query=query,
            ground_truth_answer=ground_truth,
            relevant_doc_ids=relevant_doc_ids,
            retrieved_doc_ids=retrieved_ids,
            generated_answer=answer,
            retrieved_contexts=contexts,
        ))

    return eval_samples


def _sample_value(sample: Any, key: str, default: Any) -> Any:
    if isinstance(sample, dict):
        return sample.get(key, default)
    return getattr(sample, key, default)


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _parse_score_json(text: str) -> dict[str, Any] | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def _clamp_score(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _eval_context(item: Any) -> str:
    return "\n".join([
        f"Name: {item.food_name}",
        f"Cuisine: {item.cuisine_type}",
        f"Calories: {item.calories}",
        f"Description: {item.description}",
        f"Ingredients: {item.ingredients}",
        f"Nutrition: {item.nutrition}",
        f"Health Benefits: {item.health_benefits}",
        f"Taste Profile: {item.taste_profile}",
        f"Cooking Method: {item.cooking_method}",
    ])
