from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .rag_pipeline import GeminiClient

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
    k: int = 10


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
        "faithfulness": 0.90,
        "answer_relevancy": 0.85,
        "context_precision": 0.70,
        "context_recall": 0.80,
        "answer_correctness": 0.70,
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


def evaluate_retrieval(samples: list[EvalSample], k: int = 10) -> RetrievalMetrics:
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
        precisions.append(len(hits_in_top_k) / k if k > 0 else 0.0)
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


def evaluate_generation(
    samples: list[EvalSample],
    gemini_client: "GeminiClient",
) -> GenerationMetrics:
    if not samples:
        return GenerationMetrics()

    faithfulness_scores: list[float] = []
    relevancy_scores: list[float] = []
    ctx_precision_scores: list[float] = []
    ctx_recall_scores: list[float] = []
    correctness_scores: list[float] = []

    for sample in samples:
        context_text = "\n\n".join(sample.retrieved_contexts[:5])

        faithfulness_scores.append(_llm_score(gemini_client, _faithfulness_prompt(
            sample.query, sample.generated_answer, context_text
        )))
        relevancy_scores.append(_llm_score(gemini_client, _relevancy_prompt(
            sample.query, sample.generated_answer
        )))
        ctx_precision_scores.append(_llm_score(gemini_client, _context_precision_prompt(
            sample.query, context_text
        )))
        ctx_recall_scores.append(_llm_score(gemini_client, _context_recall_prompt(
            sample.ground_truth_answer, context_text
        )))
        correctness_scores.append(_llm_score(gemini_client, _correctness_prompt(
            sample.query, sample.generated_answer, sample.ground_truth_answer
        )))

    def safe_mean(lst: list[float]) -> float:
        return statistics.mean(lst) if lst else 0.0

    return GenerationMetrics(
        faithfulness=safe_mean(faithfulness_scores),
        answer_relevancy=safe_mean(relevancy_scores),
        context_precision=safe_mean(ctx_precision_scores),
        context_recall=safe_mean(ctx_recall_scores),
        answer_correctness=safe_mean(correctness_scores),
    )


def run_full_eval(
    samples: list[EvalSample],
    gemini_client: "GeminiClient",
    k: int = 10,
) -> EvalReport:
    retrieval = evaluate_retrieval(samples, k=k)
    generation = evaluate_generation(samples, gemini_client)
    report = EvalReport(
        n_samples=len(samples),
        retrieval=retrieval,
        generation=generation,
    )
    report.compute_pass_fail()
    return report


def _llm_score(client: "GeminiClient", prompt: str) -> float:
    try:
        import re
        raw = client.generate(prompt).strip()
        match = re.search(r"\b([01](?:\.\d+)?)\b", raw)
        if match:
            return max(0.0, min(1.0, float(match.group(1))))
    except Exception as exc:
        logger.warning("LLM judge call failed: %s", exc)
    return 0.0


def _faithfulness_prompt(query: str, answer: str, context: str) -> str:
    return f"""Rate whether every claim in the ANSWER is supported by the CONTEXT.
Score: 1.0 = fully grounded, 0.0 = completely hallucinated. Decimals allowed.
Respond with a single decimal number between 0.0 and 1.0.

QUERY: {query}
CONTEXT: {context[:2000]}
ANSWER: {answer}
SCORE:"""


def _relevancy_prompt(query: str, answer: str) -> str:
    return f"""Rate how well the ANSWER addresses the QUERY.
Score: 1.0 = perfectly addresses it, 0.0 = completely off-topic. Decimals allowed.
Respond with a single decimal number between 0.0 and 1.0.

QUERY: {query}
ANSWER: {answer}
SCORE:"""


def _context_precision_prompt(query: str, context: str) -> str:
    return f"""Rate what fraction of the retrieved CONTEXT is actually relevant to the QUERY.
Score: 1.0 = all context is relevant, 0.0 = none is relevant. Decimals allowed.
Respond with a single decimal number between 0.0 and 1.0.

QUERY: {query}
CONTEXT: {context[:2000]}
SCORE:"""


def _context_recall_prompt(ground_truth: str, context: str) -> str:
    return f"""Rate how much of the GROUND TRUTH information is present in the CONTEXT.
Score: 1.0 = all ground truth info is in context, 0.0 = none. Decimals allowed.
Respond with a single decimal number between 0.0 and 1.0.

GROUND TRUTH: {ground_truth}
CONTEXT: {context[:2000]}
SCORE:"""


def _correctness_prompt(query: str, answer: str, ground_truth: str) -> str:
    return f"""Rate how correct and complete the ANSWER is compared to the GROUND TRUTH.
Score: 1.0 = fully correct, 0.0 = completely wrong. Decimals allowed.
Respond with a single decimal number between 0.0 and 1.0.

QUERY: {query}
GROUND TRUTH: {ground_truth}
ANSWER: {answer}
SCORE:"""
