from __future__ import annotations

import logging
import re
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
) -> TransformedQuery:
    route = _route_query(query, llm_client)

    if route != QueryRoute.RETRIEVAL:
        return TransformedQuery(
            original=query,
            variants=[query],
            stepback_query=None,
            route=route,
        )

    variants = _multi_query(query, llm_client, n_variants)
    stepback = _stepback(query, llm_client) if use_stepback else None

    return TransformedQuery(
        original=query,
        variants=variants,
        stepback_query=stepback,
        route=route,
    )


def _route_query(query: str, client: "OllamaClient") -> QueryRoute:
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
        logger.warning("Router failed (%s), defaulting to RETRIEVAL", exc)

    return QueryRoute.RETRIEVAL


def _multi_query(query: str, client: "OllamaClient", n: int) -> list[str]:
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
        logger.warning("Multi-query generation failed (%s)", exc)
        return [query]


def _stepback(query: str, client: "OllamaClient") -> str | None:
    prompt = f"""Given this food query, generate a broader, more general version
that could help find relevant results even if the exact dish isn't available.
Output only the broader query. No explanation.

Specific query: {query}
Broader query:"""

    try:
        return client.generate(prompt).strip()
    except Exception as exc:
        logger.warning("Step-back generation failed (%s)", exc)
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
