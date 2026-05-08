from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class QueryFilters:
    cuisine: str | None = None
    max_calories: int | None = None
    min_calories: int | None = None
    dietary_tags: list[str] = field(default_factory=list)
    allergens_to_avoid: list[str] = field(default_factory=list)
    mood_keywords: list[str] = field(default_factory=list)
    servings: int | None = None


CUISINE_KEYWORDS: dict[str, str] = {
    "american": "American",
    "chinese": "Chinese",
    "french": "French",
    "greek": "Greek",
    "indian": "Indian",
    "italian": "Italian",
    "japanese": "Japanese",
    "korean": "Korean",
    "mediterranean": "Mediterranean",
    "mexican": "Mexican",
    "spanish": "Spanish",
    "thai": "Thai",
}

DIETARY_KEYWORDS: dict[str, str] = {
    "diabetic friendly": "diabetic_friendly",
    "diabetic-friendly": "diabetic_friendly",
    "dairy free": "dairy_free",
    "dairy-free": "dairy_free",
    "fiber rich": "high_fiber",
    "fiber-rich": "high_fiber",
    "gluten free": "gluten_free",
    "gluten-free": "gluten_free",
    "heart healthy": "heart_healthy",
    "heart-healthy": "heart_healthy",
    "high fiber": "high_fiber",
    "high-fiber": "high_fiber",
    "high protein": "high_protein",
    "high-protein": "high_protein",
    "iron rich": "iron_rich",
    "iron-rich": "iron_rich",
    "keto": "keto_friendly",
    "keto friendly": "keto_friendly",
    "keto-friendly": "keto_friendly",
    "low carb": "low_carb",
    "low-carb": "low_carb",
    "low fat": "low_fat",
    "low-fat": "low_fat",
    "low sodium": "low_sodium",
    "low-sodium": "low_sodium",
    "muscle recovery": "muscle_recovery",
    "muscle-recovery": "muscle_recovery",
    "paleo": "paleo",
    "potassium rich": "potassium_rich",
    "potassium-rich": "potassium_rich",
    "protein-rich": "high_protein",
    "rich in fiber": "high_fiber",
    "rich in iron": "iron_rich",
    "rich in potassium": "potassium_rich",
    "rich in protein": "high_protein",
    "vegan": "vegan",
    "vegetarian": "vegetarian",
}

ALLERGEN_KEYWORDS: tuple[str, ...] = (
    "dairy", "egg", "eggs", "fish", "gluten", "nut", "nuts",
    "peanut", "peanuts", "sesame", "shellfish", "soy", "wheat",
)

MOOD_KEYWORDS: tuple[str, ...] = (
    "comfort", "creamy", "crispy", "fresh", "hearty", "heavy",
    "healthy", "light", "mild", "refreshing", "savory", "savoury",
    "spicy", "sweet",
)


def parse_query(query: str) -> QueryFilters:
    normalized = query.lower()
    min_calories, max_calories = _extract_calorie_range(normalized)
    return QueryFilters(
        cuisine=_extract_cuisine(normalized),
        min_calories=min_calories,
        max_calories=max_calories,
        dietary_tags=_extract_dietary_tags(normalized),
        allergens_to_avoid=_extract_allergens(normalized),
        mood_keywords=_extract_mood_keywords(normalized),
        servings=_extract_servings(normalized),
    )


def _extract_cuisine(text: str) -> str | None:
    for keyword, canonical in CUISINE_KEYWORDS.items():
        if re.search(rf"\b{re.escape(keyword)}\b", text):
            return canonical
    return None


def _extract_calorie_range(text: str) -> tuple[int | None, int | None]:
    between_match = re.search(
        r"(?:between|from)\s+(\d+)\s*(?:and|to|-)\s*(\d+)\s*(?:calories|calorie|cal|kcal)?",
        text,
    )
    around_match = re.search(
        r"(?:around|about|near)\s+(\d+)\s*(?:calories|calorie|cal|kcal)?",
        text,
    )
    under_match = re.search(
        r"(?:under|below|less than|fewer than|up to|max(?:imum)?)\s+(\d+)\s*(?:calories|calorie|cal|kcal)?",
        text,
    )
    above_match = re.search(
        r"(?:over|above|more than|at least|min(?:imum)?)\s+(\d+)\s*(?:calories|calorie|cal|kcal)?",
        text,
    )

    if between_match:
        lower = int(between_match.group(1))
        upper = int(between_match.group(2))
        return min(lower, upper), max(lower, upper)
    if around_match:
        midpoint = int(around_match.group(1))
        return max(0, midpoint - 100), midpoint + 100

    min_calories = int(above_match.group(1)) if above_match else None
    max_calories = int(under_match.group(1)) if under_match else None
    return min_calories, max_calories


def _extract_dietary_tags(text: str) -> list[str]:
    tags: list[str] = []
    for phrase, tag in DIETARY_KEYWORDS.items():
        if phrase in text and tag not in tags:
            tags.append(tag)
    return tags


def _extract_allergens(text: str) -> list[str]:
    allergens: list[str] = []
    for allergen in ALLERGEN_KEYWORDS:
        patterns = (
            rf"\b(?:no|without|avoid|free from)\s+(?:\w+\s+){{0,3}}{re.escape(allergen)}\b",
            rf"\b{re.escape(allergen)}[- ]free\b",
        )
        if any(re.search(pattern, text) for pattern in patterns):
            canonical = _canonical_allergen(allergen)
            if canonical not in allergens:
                allergens.append(canonical)
    return allergens


def _canonical_allergen(allergen: str) -> str:
    if allergen in {"nuts", "peanuts"}:
        return "nut"
    if allergen == "eggs":
        return "egg"
    return allergen


def _extract_mood_keywords(text: str) -> list[str]:
    return [
        keyword
        for keyword in MOOD_KEYWORDS
        if re.search(rf"\b{re.escape(keyword)}\b", text)
    ]


def _extract_servings(text: str) -> int | None:
    serving_match = re.search(
        r"(?:for|serves?|serving)\s+(\d+)\s*(?:people|persons?|servings?)?",
        text,
    )
    if not serving_match:
        return None
    return int(serving_match.group(1))
