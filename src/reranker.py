from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from .filters import QueryFilters
from .utils import safe_int

logger = logging.getLogger(__name__)

_VECTOR_WEIGHT = 0.6
_META_WEIGHT = 0.4


@dataclass
class FoodResult:
    food_id: str
    food_name: str
    cuisine_type: str
    calories: int
    description: str
    ingredients: str
    nutrition: str
    health_benefits: str
    taste_profile: str
    cooking_method: str
    raw_similarity: float
    cross_encoder_score: float = 0.0
    rerank_score: float = 0.0


def normalize_search_result(raw: dict[str, Any]) -> FoodResult:
    similarity = max(0.0, min(1.0, raw.get("similarity_score", 0.0) / 100.0))
    return FoodResult(
        food_id=str(raw.get("food_id", "")),
        food_name=str(raw.get("food_name", "Unknown")),
        cuisine_type=str(raw.get("cuisine_type", "Unknown")),
        calories=safe_int(raw.get("food_calories_per_serving"), 0),
        description=str(raw.get("food_description", "")),
        ingredients=str(raw.get("food_ingredients", "")),
        nutrition=str(raw.get("food_nutritional_factors", "")),
        health_benefits=str(raw.get("food_health_benefits", "")),
        taste_profile=str(raw.get("taste_profile", "")),
        cooking_method=str(raw.get("cooking_method", "")),
        raw_similarity=similarity,
    )


class CrossEncoderReranker:
    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or self.DEFAULT_MODEL
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self._model_name)
                logger.info("Cross-encoder loaded: %s", self._model_name)
            except ImportError:
                logger.warning("sentence_transformers not installed. Cross-encoder disabled.")
                self._model = None
        return self._model

    def score(self, query: str, results: list[FoodResult]) -> list[FoodResult]:
        model = self._load()
        if model is None:
            return results

        pairs = [(query, _result_text(r)) for r in results]
        try:
            import math
            raw_scores = model.predict(pairs)
            for result, score in zip(results, raw_scores):
                result.cross_encoder_score = 1.0 / (1.0 + math.exp(-float(score)))
        except Exception as exc:
            logger.warning("Cross-encoder scoring failed: %s", exc)

        return results


def rerank_results(
    results: list[FoodResult],
    filters: QueryFilters,
    query: str = "",
    cross_encoder: CrossEncoderReranker | None = None,
) -> list[FoodResult]:
    if cross_encoder is not None and query:
        results = cross_encoder.score(query, results)

    for result in results:
        metadata_score = _metadata_score(result, filters)
        allergen_penalty = _allergen_penalty(result, filters)

        if result.cross_encoder_score > 0:
            base = 0.5 * result.cross_encoder_score + 0.3 * result.raw_similarity + 0.2 * metadata_score
        else:
            base = _VECTOR_WEIGHT * result.raw_similarity + _META_WEIGHT * metadata_score

        result.rerank_score = max(0.0, min(1.0, base - allergen_penalty))

    results.sort(key=lambda r: r.rerank_score, reverse=True)

    return _mmr_select(results, lambda_param=0.6, top_k=len(results))


def _mmr_select(
    results: list[FoodResult],
    lambda_param: float = 0.6,
    top_k: int | None = None,
) -> list[FoodResult]:
    if not results:
        return results

    n = top_k or len(results)
    selected: list[FoodResult] = []
    remaining = list(results)

    while remaining and len(selected) < n:
        if not selected:
            selected.append(remaining.pop(0))
            continue

        best_idx = 0
        best_mmr = float("-inf")

        for i, candidate in enumerate(remaining):
            rel_score = candidate.rerank_score
            max_overlap = max(_text_overlap(candidate, sel) for sel in selected)
            mmr = lambda_param * rel_score - (1 - lambda_param) * max_overlap
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = i

        selected.append(remaining.pop(best_idx))

    return selected


def _text_overlap(a: FoodResult, b: FoodResult) -> float:
    tokens_a = set(re.findall(r"\b\w+\b", a.ingredients.lower()))
    tokens_b = set(re.findall(r"\b\w+\b", b.ingredients.lower()))
    if not tokens_a and not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union) if union else 0.0


def _metadata_score(result: FoodResult, filters: QueryFilters) -> float:
    score = 0.0
    combined = _combined_text(result)

    if filters.cuisine and result.cuisine_type.lower() == filters.cuisine.lower():
        score += 0.25
    if filters.max_calories is not None and result.calories <= filters.max_calories:
        score += 0.20
    if filters.min_calories is not None and result.calories >= filters.min_calories:
        score += 0.15

    dietary_hits = sum(1 for tag in filters.dietary_tags if tag.replace("_", " ") in combined)
    score += min(dietary_hits * 0.10, 0.20)

    mood_hits = sum(1 for kw in filters.mood_keywords if kw in combined)
    score += min(mood_hits * 0.05, 0.20)

    return max(0.0, min(1.0, score))


def _allergen_penalty(result: FoodResult, filters: QueryFilters) -> float:
    combined = _combined_text(result)
    penalty = 0.0
    for allergen in filters.allergens_to_avoid:
        if _contains_allergen(combined, allergen):
            penalty += 0.30
    return min(0.9, penalty)


def _combined_text(result: FoodResult) -> str:
    return " ".join([
        result.description, result.ingredients, result.nutrition,
        result.health_benefits, result.taste_profile, result.cooking_method,
    ]).lower()


def _result_text(result: FoodResult) -> str:
    return (
        f"{result.food_name}. {result.description}. "
        f"Ingredients: {result.ingredients}. "
        f"Cuisine: {result.cuisine_type}. Calories: {result.calories}. "
        f"Taste: {result.taste_profile}."
    )


def _contains_allergen(text: str, allergen: str) -> bool:
    synonyms = {
        "dairy": ("dairy", "milk", "cream", "cheese", "butter", "yogurt"),
        "egg": ("egg", "eggs"),
        "fish": ("fish",),
        "gluten": ("gluten", "wheat", "flour"),
        "nut": ("nut", "nuts", "peanut", "peanuts", "almond", "almonds",
                "walnut", "walnuts", "cashew", "cashews", "pecan", "pecans",
                "hazelnut", "hazelnuts", "pistachio", "pistachios"),
        "sesame": ("sesame",),
        "shellfish": ("shellfish", "shrimp", "prawn", "crab", "lobster"),
        "soy": ("soy", "soya", "tofu"),
        "wheat": ("wheat", "flour"),
    }
    terms = synonyms.get(allergen, (allergen,))
    return any(re.search(rf"\b{re.escape(t)}\b", text) for t in terms)
