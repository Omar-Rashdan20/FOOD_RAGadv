from __future__ import annotations

import logging
import json
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

from .cache import HybridCache, TTLCache, make_cache_key
from .config import Settings, get_settings
from .data_loader import load_food_data
from .filters import QueryFilters, parse_query
from .hybrid_retriever import BM25Index, hybrid_retrieve
from .query_transformer import QueryRoute, TransformedQuery, route_query, transform_query
from .reranker import (
    CrossEncoderReranker,
    FoodResult,
    normalize_search_result,
    rerank_results,
)
from .vector_store import (
    build_document_text,
    create_collection,
    create_embedding_function,
    get_chroma_client,
    populate_collection,
)

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model_name: str,
        max_output_tokens: int = 1024,
        temperature: float = 0.5,
        timeout_seconds: int = 120,
        max_retries: int = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, max_retries)

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_output_tokens,
            },
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        last_exc: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    data = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.URLError as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        "Ollama request failed. Make sure Ollama is running at "
                        f"{self.base_url} and the model is available: ollama pull {self.model_name}"
                    ) from exc
                time.sleep(0.4 * (2 ** attempt))
        else:
            raise RuntimeError("Ollama request failed") from last_exc

        if data.get("error"):
            raise RuntimeError(f"Ollama error: {data['error']}")
        return str(data.get("response", "")).strip()


class FoodRAGPipeline:
    def __init__(
        self,
        collection: Any,
        bm25_index: BM25Index,
        llm_client: OllamaClient,
        cross_encoder: CrossEncoderReranker,
        cache: HybridCache,
        settings: Settings,
    ) -> None:
        self.collection = collection
        self.bm25_index = bm25_index
        self.llm_client = llm_client
        self.cross_encoder = cross_encoder
        self.cache = cache
        self.settings = settings
        self.route_cache = TTLCache(ttl_seconds=settings.cache_ttl_seconds)

    def clear_caches(self) -> None:
        self.cache.clear()
        self.route_cache.clear()

    def rag_recommend(
        self,
        query: str,
        n_results: int | None = None,
        use_cache: bool = True,
        request_id: str | None = None,
    ) -> str:
        request_id = request_id or uuid.uuid4().hex[:12]
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("Query must not be empty.")

        result_count = n_results or self.settings.default_n_results
        filters = parse_query(clean_query)
        cache_key = make_cache_key(clean_query, result_count, filters)

        if use_cache:
            cached = self.cache.get(cache_key, query=clean_query)
            if cached:
                logger.info("Recommendation cache hit request_id=%s", request_id)
                return cached

        route_cache_key = f"route:{clean_query.lower()}"
        cached_route = self.route_cache.get(route_cache_key) if use_cache else None
        route = QueryRoute(cached_route) if cached_route else route_query(
            clean_query,
            self.llm_client,
            request_id=request_id,
        )
        if use_cache and cached_route is None:
            self.route_cache.set(route_cache_key, route.value)

        transformed = transform_query(
            clean_query,
            self.llm_client,
            n_variants=4,
            use_stepback=True,
            initial_route=route,
            request_id=request_id,
        )

        if transformed.route == QueryRoute.CLARIFICATION:
            return (
                "Could you give me a bit more detail? For example: "
                "what cuisine, dietary preference, or calorie range are you looking for?"
            )
        if transformed.route == QueryRoute.REJECTION:
            return (
                "I'm a food recommendation assistant. "
                "I can help you find dishes, recipes, and nutrition information. "
                "Please ask me something food-related!"
            )
        if transformed.route == QueryRoute.GENERATION:
            prompt = (
                f"You are a knowledgeable food assistant. "
                f"Answer this food-related question:\n\n{clean_query}"
            )
            try:
                return self.llm_client.generate(prompt)
            except Exception as exc:
                logger.warning("Generation route failed request_id=%s: %s", request_id, exc)
                return f"Generation error: {exc}"

        candidate_pool = self._retrieve_candidates(transformed, filters, result_count)

        if not candidate_pool:
            return (
                "I could not find food options matching your request. "
                "Try broadening the cuisine, calorie, or dietary constraints."
            )

        food_results = [normalize_search_result(r) for r in candidate_pool]
        ranked = rerank_results(
            food_results,
            filters,
            query=clean_query,
            cross_encoder=self.cross_encoder,
        )

        context = prepare_context(ranked, top_k=result_count)
        prompt = build_prompt(clean_query, context, filters, transformed)
        try:
            response = self.llm_client.generate(prompt)
        except Exception as exc:
            logger.warning("Grounded generation failed request_id=%s: %s", request_id, exc)
            return f"Generation error: {exc}"

        if response and use_cache:
            self.cache.set(cache_key, query=clean_query, value=response)

        return response

    def _retrieve_candidates(
        self,
        transformed: TransformedQuery,
        filters: QueryFilters,
        n_results: int,
    ) -> list[dict[str, Any]]:
        try:
            total = self.collection.count()
        except Exception:
            total = 100

        fetch_n = max(1, min(total, n_results * self.settings.retrieval_multiplier))

        return hybrid_retrieve(
            collection=self.collection,
            queries=transformed.variants,
            filters=filters,
            n_results=fetch_n,
            bm25_index=self.bm25_index,
            confidence_threshold=self.settings.confidence_threshold,
        )


def prepare_context(results: list[FoodResult], top_k: int = 3) -> str:
    if not results:
        return "No relevant food items were found."

    sections: list[str] = []
    for rank, item in enumerate(results[:top_k], start=1):
        sections.append("\n".join([
            f"[OPTION {rank}]",
            f"Name: {item.food_name}",
            f"Cuisine: {item.cuisine_type}",
            f"Calories: {item.calories} per serving",
            f"Match Score: {item.rerank_score * 100:.1f}%",
            f"Description: {item.description}",
            f"Ingredients: {item.ingredients}",
            f"Nutrition: {item.nutrition}",
            f"Health Benefits: {item.health_benefits}",
            f"Taste Profile: {item.taste_profile}",
            f"Cooking Method: {item.cooking_method}",
        ]))
    return "\n\n".join(sections)


def build_prompt(
    query: str,
    context: str,
    filters: QueryFilters,
    transformed: TransformedQuery | None = None,
) -> str:
    filter_block = _format_filters(filters)
    stepback_block = ""
    if transformed and transformed.stepback_query:
        stepback_block = f"\nBroader context question considered: {transformed.stepback_query}"

    return f"""You are an expert food recommendation assistant.

Strict rules:
- Recommend ONLY foods from the retrieved options below.
- NEVER invent dishes or facts not present in the retrieved context.
- Respect ALL cuisine, calorie, dietary, and allergen constraints.
- If no option fully matches, say so honestly and offer the closest matches.

Example request: "light Italian pasta under 500 calories"
Example response:
- **Spaghetti Aglio e Olio** (Italian, 420 cal) fits the calorie target, garlic-forward flavor.
- **Pasta Primavera** (Italian, 390 cal) is vegetable-heavy and lighter than cream-based pasta.

User request: {query}{stepback_block}

Detected filters:
{filter_block}

Retrieved options:
{context}

Write a concise, grounded answer:
### Response format
Respond only with a markdown bullet list.
Use 2-3 bullets.
Each bullet must start exactly like:
- **Name** (Cuisine, calories): why it fits.
If a constraint cannot be fully satisfied, add one final bullet:
- **Note**: brief limitation.
"""


def _format_filters(filters: QueryFilters) -> str:
    parts: list[str] = []
    if filters.cuisine:
        parts.append(f"- Cuisine: {filters.cuisine}")
    if filters.min_calories is not None:
        parts.append(f"- Min calories: {filters.min_calories}")
    if filters.max_calories is not None:
        parts.append(f"- Max calories: {filters.max_calories}")
    if filters.dietary_tags:
        parts.append(f"- Dietary tags: {', '.join(filters.dietary_tags)}")
    if filters.allergens_to_avoid:
        parts.append(f"- Avoid allergens: {', '.join(filters.allergens_to_avoid)}")
    if filters.mood_keywords:
        parts.append(f"- Taste or mood: {', '.join(filters.mood_keywords)}")
    if filters.servings:
        parts.append(f"- Servings: {filters.servings}")
    return "\n".join(parts) if parts else "- None detected"


def build_pipeline(
    settings: Settings | None = None,
    rebuild_index: bool = False,
    enable_cross_encoder: bool = True,
    cross_encoder_model: str | None = None,
) -> FoodRAGPipeline:
    app_settings = settings or get_settings()

    food_items = load_food_data(app_settings.dataset_path)

    embedding_function = create_embedding_function(app_settings.embedding_model_name)
    client = get_chroma_client(app_settings.chroma_db_path)
    collection = create_collection(
        client=client,
        collection_name=app_settings.collection_name,
        embedding_function=embedding_function,
        metadata={"description": "Food search collection — hybrid RAG"},
        reset=rebuild_index,
    )

    for food in food_items:
        food["document"] = build_document_text(food)

    populate_collection(
        collection=collection,
        food_items=food_items,
        batch_size=app_settings.batch_size,
    )

    bm25 = BM25Index()
    bm25.build(food_items)

    cross_encoder = CrossEncoderReranker(model_name=cross_encoder_model) if enable_cross_encoder else None

    def _embed_fn(texts: list[str]) -> Any:
        return embedding_function(texts)

    cache = HybridCache(
        ttl_seconds=app_settings.cache_ttl_seconds,
        embedding_fn=_embed_fn,
        semantic_threshold=0.92,
    )

    llm_client = OllamaClient(
        base_url=app_settings.ollama_base_url,
        model_name=app_settings.ollama_model_name,
        max_output_tokens=app_settings.max_output_tokens,
        temperature=app_settings.temperature,
        timeout_seconds=app_settings.ollama_timeout_seconds,
        max_retries=app_settings.ollama_max_retries,
    )

    return FoodRAGPipeline(
        collection=collection,
        bm25_index=bm25,
        llm_client=llm_client,
        cross_encoder=cross_encoder or CrossEncoderReranker(),
        cache=cache,
        settings=app_settings,
    )
