from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .rag_pipeline import OllamaClient

logger = logging.getLogger(__name__)


class QueryRoute(str, Enum):
    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    CLARIFICATION = "clarification"
    REJECTION = "rejection"


@dataclass
class TransformedQuery:
    original: str
    variants: list[str]
    stepback_query: str | None
    route: QueryRoute


def transform_query(
    query: str,
    llm_client: "OllamaClient",
    *,
    n_variants: int = 4,
    use_stepback: bool = True,
    initial_route: QueryRoute | None = None,
    request_id: str | None = None,
) -> TransformedQuery:
    route = initial_route or _route_query(query, llm_client, request_id=request_id)

    if route != QueryRoute.RETRIEVAL:
        return TransformedQuery(
            original=query,
            variants=[query],
            stepback_query=None,
            route=route,
        )

    if use_stepback:
        with ThreadPoolExecutor(max_workers=2) as executor:
            variants_future = executor.submit(_multi_query, query, llm_client, n_variants, request_id)
            stepback_future = executor.submit(_stepback, query, llm_client, request_id)
            variants = variants_future.result()
            stepback = stepback_future.result()
    else:
        variants = _multi_query(query, llm_client, n_variants, request_id=request_id)
        stepback = None

    return TransformedQuery(
        original=query,
        variants=variants,
        stepback_query=stepback,
        route=route,
    )


def route_query(
    query: str,
    client: "OllamaClient",
    *,
    request_id: str | None = None,
) -> QueryRoute:
    return _route_query(query, client, request_id=request_id)


def _route_query(
    query: str,
    client: "OllamaClient",
    *,
    request_id: str | None = None,
) -> QueryRoute:
    regex_route = _regex_route(query)
    if regex_route is not None:
        return regex_route

    prompt = f"""You are a query router for a food recommendation system.
Classify the user query into exactly ONE category:

- RETRIEVAL   : Needs food/recipe search (specific dish, cuisine, ingredient, diet, calories)
- GENERATION  : Creative or analytical request that doesn't need retrieval
- CLARIFICATION: Too vague to process
- REJECTION   : Completely out of scope (not food related)

Respond with only the single word: RETRIEVAL, GENERATION, CLARIFICATION, or REJECTION.

Query: {query}"""

    try:
        raw = client.generate(prompt).strip().upper()
        for route in QueryRoute:
            if route.value.upper() in raw:
                return route
    except Exception as exc:
        logger.warning("Router failed request_id=%s (%s), defaulting to RETRIEVAL", request_id, exc)

    return QueryRoute.RETRIEVAL


def _multi_query(
    query: str,
    client: "OllamaClient",
    n: int,
    request_id: str | None = None,
) -> list[str]:
    prompt = f"""You are helping improve a food search engine.
Generate {n - 1} alternative phrasings of the user's food query.
Each variant should cover a different semantic angle or keyword perspective.
Output ONLY the variants as a numbered list (1. ... 2. ... etc.). No extra text.

Original query: {query}"""

    try:
        raw = client.generate(prompt)
        variants = _parse_numbered_list(raw)
        seen = {query.lower()}
        unique = [query]
        for v in variants:
            if v.lower() not in seen:
                seen.add(v.lower())
                unique.append(v)
        return unique[:n]
    except Exception as exc:
        logger.warning("Multi-query generation failed request_id=%s (%s)", request_id, exc)
        return [query]


def _stepback(
    query: str,
    client: "OllamaClient",
    request_id: str | None = None,
) -> str | None:
    prompt = f"""Given this food query, generate a broader, more general version
that could help find relevant results even if the exact dish isn't available.
Output only the broader query. No explanation.

Specific query: {query}
Broader query:"""

    try:
        return client.generate(prompt).strip()
    except Exception as exc:
        logger.warning("Step-back generation failed request_id=%s (%s)", request_id, exc)
        return None


def _parse_numbered_list(text: str) -> list[str]:
    lines = text.strip().splitlines()
    results: list[str] = []
    for line in lines:
        line = line.strip()
        cleaned = re.sub(r"^\d+[.)]\s*", "", line).strip()
        cleaned = re.sub(r"^[-*]\s*", "", cleaned).strip()
        if cleaned:
            results.append(cleaned)
    return results


def _regex_route(query: str) -> QueryRoute | None:
    text = query.strip().lower()
    if len(text.split()) <= 1 and not re.search(r"\b(food|meal|recipe|calorie|protein|fat|carb|fiber)\b", text):
        return QueryRoute.CLARIFICATION

    food_signals = (
        "food", "meal", "recipe", "diet", "calorie", "protein", "fat", "carb",
        "fiber", "sodium", "iron", "potassium", "vegan", "vegetarian", "keto",
        "breakfast", "lunch", "dinner", "snack", "healthy",
    )
    obvious_non_food = (
        "weather", "stock", "bitcoin", "football", "movie", "code", "python",
        "javascript", "travel", "hotel", "car", "phone",
    )
    if any(term in text for term in obvious_non_food) and not any(term in text for term in food_signals):
        return QueryRoute.REJECTION

    return None
