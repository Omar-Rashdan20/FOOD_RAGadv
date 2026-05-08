from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_food_data(path: str) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        logger.warning("Dataset not found at %s — returning empty list.", path)
        return []

    with file_path.open(encoding="utf-8") as f:
        raw = json.load(f)

    items = raw if isinstance(raw, list) else raw.get("foods", raw.get("data", []))
    normalized = [_normalize(item, idx) for idx, item in enumerate(items)]
    logger.info("Loaded %d food items from %s", len(normalized), path)
    return normalized


def _normalize(item: dict[str, Any], idx: int) -> dict[str, Any]:
    ingredients = item.get("food_ingredients", item.get("ingredients", []))
    if isinstance(ingredients, list):
        ingredients = ", ".join(str(i) for i in ingredients)

    return {
        "food_id": str(item.get("food_id", item.get("id", idx))),
        "food_name": str(item.get("food_name", item.get("name", "Unknown"))),
        "food_description": str(item.get("food_description", item.get("description", ""))),
        "food_ingredients": ingredients,
        "food_calories_per_serving": _safe_int(item.get("food_calories_per_serving", item.get("calories", 0))),
        "cuisine_type": str(item.get("cuisine_type", item.get("cuisine", "Unknown"))),
        "cooking_method": str(item.get("cooking_method", "")),
        "taste_profile": str(item.get("taste_profile", "")),
        "food_health_benefits": str(item.get("food_health_benefits", item.get("health_benefits", ""))),
        "nutrition_profile": str(item.get("nutrition_profile", item.get("food_nutritional_factors", ""))),
        "dietary_tags": item.get("dietary_tags", []),
        "allergens": item.get("allergens", []),
    }


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
