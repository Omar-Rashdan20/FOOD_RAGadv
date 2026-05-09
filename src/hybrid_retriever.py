from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any

from .filters import QueryFilters
from .vector_store import build_where_filter, format_results, query_collection

logger = logging.getLogger(__name__)

_RRF_K = 60


def hybrid_retrieve(
    collection: Any,
    queries: list[str],
    filters: QueryFilters,
    n_results: int,
    bm25_index: "BM25Index | None" = None,
    confidence_threshold: float | None = None,
) -> list[dict[str, Any]]:
    where = build_where_filter(filters)

    all_dense_lists: list[list[dict[str, Any]]] = []
    all_sparse_lists: list[list[dict[str, Any]]] = []

    for query in queries:
        raw = query_collection(
            collection,
            query=query,
            n_results=n_results * 3,
            where=where,
        )
        dense_results = format_results(raw)
        all_dense_lists.append(dense_results)

        if bm25_index is not None:
            sparse_results = bm25_index.search(query, n_results=n_results * 3, filters=filters)
            all_sparse_lists.append(sparse_results)

    fused = _rrf_fuse(all_dense_lists)

    if all_sparse_lists:
        fused = _rrf_fuse([fused] + all_sparse_lists)

    if confidence_threshold is not None:
        fused = [r for r in fused if r.get("similarity_score", 0) >= confidence_threshold]

    return fused[:n_results]


def _rrf_fuse(
    ranked_lists: list[list[dict[str, Any]]],
    k: int = _RRF_K,
) -> list[dict[str, Any]]:
    scores: dict[str, float] = defaultdict(float)
    items_by_id: dict[str, dict[str, Any]] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            food_id = str(item.get("food_id", ""))
            scores[food_id] += 1.0 / (k + rank)
            items_by_id[food_id] = item

    max_score = max(scores.values(), default=1.0)

    fused: list[dict[str, Any]] = []
    for food_id, rrf_score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        item = dict(items_by_id[food_id])
        item["similarity_score"] = round((rrf_score / max_score) * 100.0, 2)
        item["rrf_score"] = rrf_score
        fused.append(item)

    return fused


class BM25Index:
    def __init__(self) -> None:
        self._bm25: Any | None = None
        self._docs: list[dict[str, Any]] = []
        self._tokenized: list[list[str]] = []

    def build(self, food_items: list[dict[str, Any]]) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank_bm25 not installed. BM25 retrieval disabled.")
            return

        self._docs = food_items
        self._tokenized = [_tokenize(doc.get("document", "") or _doc_text(doc)) for doc in food_items]
        self._bm25 = BM25Okapi(self._tokenized)
        logger.info("BM25 index built with %d documents", len(food_items))

    def search(
        self,
        query: str,
        n_results: int = 10,
        filters: QueryFilters | None = None,
    ) -> list[dict[str, Any]]:
        if self._bm25 is None or not self._docs:
            return []

        tokenized_query = _tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:n_results * 3]

        results: list[dict[str, Any]] = []
        max_score = scores[top_indices[0]] if top_indices else 1.0
        max_score = max(max_score, 1e-9)

        for idx in top_indices:
            if scores[idx] <= 0:
                break
            doc = self._docs[idx]
            if filters is not None and not _matches_filters(doc, filters):
                continue
            normalised = round((scores[idx] / max_score) * 100.0, 2)
            results.append({
                "food_id": str(doc.get("food_id", idx)),
                "food_name": doc.get("food_name", "Unknown"),
                "food_description": doc.get("food_description", ""),
                "food_ingredients": doc.get("food_ingredients", ""),
                "food_nutritional_factors": doc.get("nutrition_profile", ""),
                "food_health_benefits": doc.get("food_health_benefits", ""),
                "taste_profile": doc.get("taste_profile", ""),
                "cooking_method": doc.get("cooking_method", ""),
                "cuisine_type": doc.get("cuisine_type", "Unknown"),
                "food_calories_per_serving": doc.get("food_calories_per_serving", 0),
                "metadata": doc.get("metadata", {}),
                "similarity_score": normalised,
                "distance": 1.0 - (scores[idx] / max_score),
                "document": doc.get("document", ""),
            })
            if len(results) >= n_results:
                break

        return results


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _doc_text(doc: dict[str, Any]) -> str:
    parts = [
        doc.get("food_name", ""),
        doc.get("food_description", ""),
        doc.get("food_ingredients", "") if isinstance(doc.get("food_ingredients"), str)
        else " ".join(doc.get("food_ingredients", [])),
        doc.get("cuisine_type", ""),
        doc.get("cooking_method", ""),
        doc.get("taste_profile", ""),
    ]
    return " ".join(p for p in parts if p)


def _matches_filters(doc: dict[str, Any], filters: QueryFilters) -> bool:
    metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
    calories = _safe_float(
        metadata.get("calories", doc.get("food_calories_per_serving"))
    )

    if filters.cuisine:
        values = {
            str(doc.get("cuisine_type", "")).lower(),
            str(metadata.get("category", "")).lower(),
        }
        if filters.cuisine.lower() not in values:
            return False
    if filters.max_calories is not None and calories is not None and calories > filters.max_calories:
        return False
    if filters.min_calories is not None and calories is not None and calories < filters.min_calories:
        return False

    for tag in filters.dietary_tags:
        if tag in _FILTER_TAGS and metadata.get(tag) is not True:
            return False

    return True


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


_FILTER_TAGS = {
    "diabetic_friendly",
    "heart_healthy",
    "high_fiber",
    "high_protein",
    "iron_rich",
    "keto_friendly",
    "low_carb",
    "low_fat",
    "low_sodium",
    "muscle_recovery",
    "potassium_rich",
    "vegan",
    "vegetarian",
}
