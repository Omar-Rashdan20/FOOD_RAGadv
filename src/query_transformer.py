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
    NORMAL_RETRIEVAL = "normal_retrieval"
    MULTI_QUERY = "multi_query"
    MULTI_QUERY_STEPBACK = "multi_query_stepback"
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

    if route in {QueryRoute.CLARIFICATION, QueryRoute.REJECTION}:
        return TransformedQuery(
            original=query,
            variants=[query],
            stepback_query=None,
            route=route,
        )

    if route == QueryRoute.NORMAL_RETRIEVAL:
        return TransformedQuery(
            original=query,
            variants=[query],
            stepback_query=None,
            route=route,
        )

    if route == QueryRoute.MULTI_QUERY:
        variants = _multi_query(query, llm_client, n_variants, request_id=request_id)
        stepback = None
    elif use_stepback:
        with ThreadPoolExecutor(max_workers=2) as executor:
            variants_future = executor.submit(_multi_query, query, llm_client, n_variants, request_id)
            stepback_future = executor.submit(_stepback, query, llm_client, request_id)
            variants = variants_future.result()
            stepback = stepback_future.result()
    else:
        variants = _multi_query(query, llm_client, n_variants, request_id=request_id)
        stepback = None

    if stepback:
        seen = {variant.lower() for variant in variants}
        if stepback.lower() not in seen:
            variants.append(stepback)

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

    prompt = f"""You are an intelligent query router for a production-grade Food RAG system.

Your job is to analyze the user query and decide which retrieval strategy should be used.

Available strategies:

1. NORMAL_RETRIEVAL
Use for:
- exact food names
- simple nutrition queries
- cuisine/category searches
- direct ingredient searches

Examples:
- "apple pie calories"
- "Italian desserts"
- "foods with chicken"

2. MULTI_QUERY
Use for:
- nutrition constraints
- calorie filtering
- diet-related searches
- ingredient combinations
- semantic food search

Examples:
- "healthy foods under 300 calories"
- "high protein low fat meals"
- "vegan foods rich in iron"

3. MULTI_QUERY_STEPBACK
Use for:
- vague semantic intent
- goal-based nutrition queries
- health outcome queries
- recommendation-style requests

Examples:
- "foods for muscle recovery"
- "foods that improve digestion"
- "healthy and filling meals"
- "foods for energy"

4. CLARIFICATION
Use when the query is too vague and lacks enough information.

Examples:
- "food"
- "healthy"
- "meal"
- "protein"

5. REJECTION
Use for queries unrelated to food, nutrition, recipes, meals, diets, or health eating.

Examples:
- "best programming language"
- "weather tomorrow"
- "football results"

IMPORTANT RULES
- Prefer NORMAL_RETRIEVAL for short and direct queries.
- Prefer MULTI_QUERY for constrained nutrition searches.
- Prefer MULTI_QUERY_STEPBACK for abstract health or recommendation queries.
- Only use CLARIFICATION when retrieval would likely fail due to lack of information.
- Use REJECTION only for clearly non-food topics.

Respond with only one strategy name:
NORMAL_RETRIEVAL, MULTI_QUERY, MULTI_QUERY_STEPBACK, CLARIFICATION, or REJECTION.

Query: {query}"""

    try:
        raw = client.generate(prompt).strip().upper()
        for route in QueryRoute:
            if route.name in raw or route.value.upper() in raw:
                return route
    except Exception as exc:
        logger.warning("Router failed request_id=%s (%s), using regex fallback", request_id, exc)

    return _regex_route(query) or QueryRoute.NORMAL_RETRIEVAL


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
    tokens = text.split()

    vague_single_terms = {
        "food", "foods", "meal", "meals", "healthy", "protein", "diet", "recipe",
        "snack", "nutrition", "calories", "calorie",
    }
    if len(tokens) <= 1 and (not tokens or tokens[0] in vague_single_terms):
        return QueryRoute.CLARIFICATION

    food_signals = (
        "food", "meal", "recipe", "diet", "calorie", "protein", "fat", "carb",
        "fiber", "sodium", "iron", "potassium", "vegan", "vegetarian", "keto",
        "breakfast", "lunch", "dinner", "snack", "healthy", "cuisine", "ingredient",
        "diabetic", "heart", "muscle", "recovery", "digestion", "energy",
    )
    obvious_non_food = (
        "weather", "stock", "bitcoin", "football", "movie", "code", "python",
        "javascript", "travel", "hotel", "car", "phone",
    )
    if any(term in text for term in obvious_non_food) and not any(term in text for term in food_signals):
        return QueryRoute.REJECTION

    goal_terms = (
        "for ", "improve", "recovery", "digestion", "energy", "filling",
        "recommend", "best", "healthy and", "suitable",
    )
    if any(term in text for term in goal_terms):
        return QueryRoute.MULTI_QUERY_STEPBACK

    if re.search(r"\bfoods?\s+with\s+[a-z0-9 ,'-]+$", text):
        return QueryRoute.NORMAL_RETRIEVAL

    constrained_patterns = (
        r"\bunder\b", r"\bover\b", r"\bless than\b", r"\bmore than\b",
        r"\blow\b", r"\bhigh\b", r"\brich in\b", r"\bwithout\b",
        r"\bvegan\b", r"\bvegetarian\b", r"\bketo\b", r"\bdiabetic\b",
        r"\bsodium\b", r"\biron\b", r"\bpotassium\b", r"\bfiber\b",
        r"\bcarbs?\b", r"\bfat\b", r"\bcalories?\b",
    )
    if any(re.search(pattern, text) for pattern in constrained_patterns):
        direct_nutrition = bool(re.search(r"\bcalories?\b$", text)) and len(tokens) <= 4
        if direct_nutrition:
            return QueryRoute.NORMAL_RETRIEVAL
        return QueryRoute.MULTI_QUERY

    if any(term in text for term in food_signals) or len(tokens) <= 4:
        return QueryRoute.NORMAL_RETRIEVAL

    return None
