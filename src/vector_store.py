from __future__ import annotations

import logging
import hashlib
import json
from typing import Any

from .filters import QueryFilters

logger = logging.getLogger(__name__)


def get_chroma_client(path: str) -> Any:
    import chromadb
    return chromadb.PersistentClient(path=path)


def create_embedding_function(model_name: str) -> Any:
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    return SentenceTransformerEmbeddingFunction(model_name=model_name)


def create_collection(
    client: Any,
    collection_name: str,
    embedding_function: Any,
    metadata: dict[str, Any] | None = None,
    reset: bool = False,
) -> Any:
    if reset:
        try:
            client.delete_collection(collection_name)
            logger.info("Deleted existing collection: %s", collection_name)
        except Exception:
            pass

    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        metadata=metadata or {},
    )


def build_document_text(food: dict[str, Any]) -> str:
    if food.get("semantic_text"):
        return str(food["semantic_text"])

    parts = [
        food.get("food_name", ""),
        food.get("food_description", ""),
        food.get("food_ingredients", ""),
        " ".join(food.get("aliases", [])) if isinstance(food.get("aliases"), list) else "",
        " ".join(food.get("tags", [])) if isinstance(food.get("tags"), list) else "",
        food.get("cuisine_type", ""),
        food.get("cooking_method", ""),
        food.get("taste_profile", ""),
        food.get("food_health_benefits", ""),
        food.get("nutrition_profile", ""),
    ]
    return " | ".join(p for p in parts if p)

def populate_collection(
    collection: Any,
    food_items: list[dict[str, Any]],
    batch_size: int = 100,
) -> None:
    existing_ids = set(collection.get(include=[])["ids"])

    seen_ids: set[str] = set()
    seen_content: set[str] = set()
    new_items: list[dict[str, Any]] = []
    for idx, f in enumerate(food_items):
        food_id = str(f["food_id"])
        content_key = _content_key(f)
        if content_key in seen_content:
            logger.debug("Skipping duplicate food content: %s", food_id)
            continue
        seen_content.add(content_key)
        if food_id in existing_ids or food_id in seen_ids:
            food_id = f"{food_id}_{content_key[:8]}"
            f = {**f, "food_id": food_id}
        seen_ids.add(food_id)
        if food_id not in existing_ids:
            new_items.append(f)

    if not new_items:
        logger.info("Collection already up to date (%d items).", len(existing_ids))
        return

    for i in range(0, len(new_items), batch_size):
        batch = new_items[i: i + batch_size]
        collection.add(
            ids=[str(f["food_id"]) for f in batch],
            documents=[f.get("document", build_document_text(f)) for f in batch],
            metadatas=[_build_metadata(f) for f in batch],
        )

    logger.info("Added %d new items to collection.", len(new_items))


def _build_metadata(food: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        "food_name": food.get("food_name", ""),
        "cuisine_type": food.get("cuisine_type", "Unknown"),
        "food_calories_per_serving": food.get("food_calories_per_serving", 0),
        "cooking_method": food.get("cooking_method", ""),
        "taste_profile": food.get("taste_profile", ""),
        "food_description": food.get("food_description", ""),
        "food_ingredients": food.get("food_ingredients", ""),
        "food_nutritional_factors": food.get("nutrition_profile", ""),
        "food_health_benefits": food.get("food_health_benefits", ""),
    }
    extra = food.get("metadata") if isinstance(food.get("metadata"), dict) else {}
    for key, value in extra.items():
        if isinstance(value, (str, int, float, bool)) and value is not None:
            metadata[key] = value
    if isinstance(food.get("tags"), list):
        metadata["tags"] = ", ".join(str(tag) for tag in food["tags"])
    return metadata


def query_collection(
    collection: Any,
    query: str,
    n_results: int = 10,
    where: dict[str, Any] | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "query_texts": [query],
        "n_results": min(n_results, max(1, collection.count())),
        "include": ["metadatas", "documents", "distances"],
    }
    if where:
        kwargs["where"] = where
    try:
        return collection.query(**kwargs)
    except Exception as exc:
        logger.warning("Collection query failed: %s", exc)
        return {"ids": [[]], "metadatas": [[]], "documents": [[]], "distances": [[]]}


def format_results(raw: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    ids = raw.get("ids", [[]])[0]
    metadatas = raw.get("metadatas", [[]])[0]
    documents = raw.get("documents", [[]])[0]
    distances = raw.get("distances", [[]])[0]

    for food_id, meta, doc, dist in zip(ids, metadatas, documents, distances):
        similarity = round(max(0.0, (1.0 - dist) * 100.0), 2)
        results.append({
            "food_id": food_id,
            "food_name": meta.get("food_name", "Unknown"),
            "food_description": meta.get("food_description", ""),
            "food_ingredients": meta.get("food_ingredients", ""),
            "food_nutritional_factors": meta.get("food_nutritional_factors", ""),
            "food_health_benefits": meta.get("food_health_benefits", ""),
            "taste_profile": meta.get("taste_profile", ""),
            "cooking_method": meta.get("cooking_method", ""),
            "cuisine_type": meta.get("cuisine_type", "Unknown"),
            "food_calories_per_serving": meta.get("food_calories_per_serving", 0),
            "metadata": meta,
            "similarity_score": similarity,
            "distance": dist,
            "document": doc,
        })

    return results


def build_where_filter(filters: QueryFilters) -> dict[str, Any] | None:
    conditions: list[dict[str, Any]] = []

    if filters.cuisine:
        conditions.append({"$or": [
            {"cuisine_type": {"$eq": filters.cuisine}},
            {"category": {"$eq": filters.cuisine}},
        ]})
    if filters.max_calories is not None:
        conditions.append({"food_calories_per_serving": {"$lte": filters.max_calories}})
    if filters.min_calories is not None:
        conditions.append({"food_calories_per_serving": {"$gte": filters.min_calories}})
    for tag in filters.dietary_tags:
        if tag in _METADATA_FILTER_TAGS:
            conditions.append({tag: {"$eq": True}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


_METADATA_FILTER_TAGS = {
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


def _content_key(food: dict[str, Any]) -> str:
    metadata = food.get("metadata") if isinstance(food.get("metadata"), dict) else {}
    payload = {
        "name": str(food.get("food_name", "")).strip().lower(),
        "calories": food.get("food_calories_per_serving", metadata.get("calories", 0)),
        "category": str(food.get("cuisine_type", metadata.get("category", ""))).strip().lower(),
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
