from __future__ import annotations

import logging
from typing import Any

from .cache import HybridCache, make_cache_key
from .config import Settings, get_settings
from .data_loader import load_food_data
from .filters import QueryFilters, parse_query
from .hybrid_retriever import BM25Index, hybrid_retrieve
from .query_transformer import QueryRoute, TransformedQuery, transform_query
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


class GeminiClient:
    def __init__(
        self,
        api_key: str | None,
        model_name: str,
        max_output_tokens: int = 1024,
        temperature: float = 0.5,
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self._client: Any | None = None
        self._types: Any | None = None

    def generate(self, prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Add it to .env before running."
            )
        client, genai_types = self._load_client()
        config = genai_types.GenerateContentConfig(
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
        )
        response = client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config,
        )
        return (response.text or "").strip()

    def _load_client(self) -> tuple[Any, Any]:
        if self._client is None or self._types is None:
            from google import genai
            from google.genai import types
            self._client = genai.Client(api_key=self.api_key)
            self._types = types
        return self._client, self._types


class FoodRAGPipeline:
    def __init__(
        self,
        collection: Any,
        bm25_index: BM25Index,
        gemini_client: GeminiClient,
        cross_encoder: CrossEncoderReranker,
        cache: HybridCache,
        settings: Settings,
    ) -> None:
        self.collection = collection
        self.bm25_index = bm25_index
        self.gemini_client = gemini_client
        self.cross_encoder = cross_encoder
        self.cache = cache
        self.settings = settings

    def rag_recommend(
        self,
        query: str,
        n_results: int | None = None,
        use_cache: bool = True,
    ) -> str:
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("Query must not be empty.")

        result_count = n_results or self.settings.default_n_results
        filters = parse_query(clean_query)
        cache_key = make_cache_key(clean_query, result_count, filters)

        if use_cache:
            cached = self.cache.get(cache_key, query=clean_query)
            if cached:
                return cached

        transformed = transform_query(
            clean_query,
            self.gemini_client,
            n_variants=4,
            use_hyde=True,
            use_stepback=True,
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
                return self.gemini_client.generate(prompt)
            except Exception as exc:
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
            response = self.gemini_client.generate(prompt)
        except Exception as exc:
            return f"Generation error: {exc}"

        if use_cache:
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
- One sentence acknowledging the request.
- Recommend 2-3 best matches with **Name** (Cuisine, calories) in bold.
- Brief explanation of why each fits.
- Mention any constraint that could NOT be fully satisfied.
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

    gemini = GeminiClient(
        api_key=app_settings.google_api_key,
        model_name=app_settings.gemini_model_name,
        max_output_tokens=app_settings.max_output_tokens,
        temperature=app_settings.temperature,
    )

    return FoodRAGPipeline(
        collection=collection,
        bm25_index=bm25,
        gemini_client=gemini,
        cross_encoder=cross_encoder or CrossEncoderReranker(),
        cache=cache,
        settings=app_settings,
    )
