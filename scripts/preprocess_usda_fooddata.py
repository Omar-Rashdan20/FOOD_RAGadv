from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


NUTRIENT_ALIASES: dict[str, tuple[str, ...]] = {
    "calories": (
        "Energy",
        "Energy (Atwater General Factors)",
        "Energy (Atwater Specific Factors)",
    ),
    "protein_g": ("Protein",),
    "fat_g": ("Total lipid (fat)",),
    "carb_g": ("Carbohydrate, by difference", "Carbohydrate, by summation"),
    "fiber_g": (
        "Fiber, total dietary",
        "Total dietary fiber (AOAC 2011.25)",
    ),
    "sodium_mg": ("Sodium, Na",),
    "iron_mg": ("Iron, Fe",),
    "potassium_mg": ("Potassium, K",),
    "calcium_mg": ("Calcium, Ca",),
    "magnesium_mg": ("Magnesium, Mg",),
    "zinc_mg": ("Zinc, Zn",),
    "sugar_g": ("Total Sugars", "Sugars, Total"),
}

LEGUME_ALIASES = {
    "small white beans": ["white beans", "navy beans", "small white beans"],
    "black beans": ["black beans", "turtle beans"],
    "chickpeas": ["chickpeas", "garbanzo beans"],
    "hummus": ["hummus", "chickpea dip"],
    "lentils": ["lentils", "dal"],
}

ANIMAL_TERMS = {
    "beef", "pork", "chicken", "turkey", "lamb", "fish", "salmon", "tuna",
    "egg", "eggs", "milk", "cheese", "yogurt", "butter", "cream", "meat",
}


def preprocess_usda_dataset(input_path: Path) -> list[dict[str, Any]]:
    raw = json.loads(input_path.read_text(encoding="utf-8"))
    foods = raw.get("FoundationFoods", raw.get("foods", raw if isinstance(raw, list) else []))

    processed: list[dict[str, Any]] = []
    seen: set[str] = set()

    for food in foods:
        if not isinstance(food, dict):
            continue

        food_name = normalize_food_name(str(food.get("description", "")))
        if not food_name:
            continue

        category = _clean_text(
            str((food.get("foodCategory") or {}).get("description", "Unknown"))
        )
        nutrients = extract_nutrients(food)
        if nutrients["calories"] is None:
            nutrients["calories"] = compute_calories(nutrients)

        dedupe_key = _dedupe_key(food_name, category, nutrients)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        tags = generate_tags(food_name, category, nutrients)
        aliases = generate_aliases(food_name, category)
        metadata = build_metadata(food, food_name, category, nutrients, tags)
        semantic_text = build_semantic_text(food_name, category, nutrients, tags, aliases)

        processed.append({
            "id": str(food.get("fdcId", food.get("ndbNumber", len(processed)))),
            "food_name": food_name,
            "aliases": aliases,
            "semantic_text": semantic_text,
            "metadata": metadata,
            "tags": tags,
        })

    return processed


def extract_nutrients(food: dict[str, Any]) -> dict[str, float | None]:
    raw_nutrients: dict[str, list[float]] = {}
    for food_nutrient in food.get("foodNutrients", []) or []:
        nutrient = food_nutrient.get("nutrient") or {}
        name = str(nutrient.get("name", "")).strip()
        unit = str(nutrient.get("unitName", "")).strip().lower()
        amount = _safe_float(food_nutrient.get("amount"))
        if not name or amount is None:
            continue

        if name == "Energy" and unit == "kj":
            amount = amount * 0.239006
        raw_nutrients.setdefault(name, []).append(amount)

    extracted: dict[str, float | None] = {}
    for field, aliases in NUTRIENT_ALIASES.items():
        extracted[field] = _first_available(raw_nutrients, aliases)

    return extracted


def compute_calories(nutrients: dict[str, float | None]) -> float | None:
    protein = nutrients.get("protein_g")
    carbs = nutrients.get("carb_g")
    fat = nutrients.get("fat_g")
    if protein is None or carbs is None or fat is None:
        return None
    return round((protein * 4) + (carbs * 4) + (fat * 9), 1)


def generate_tags(
    food_name: str,
    category: str,
    nutrients: dict[str, float | None],
) -> list[str]:
    calories = nutrients.get("calories") or 0.0
    protein = nutrients.get("protein_g") or 0.0
    fat = nutrients.get("fat_g") or 0.0
    carbs = nutrients.get("carb_g") or 0.0
    fiber = nutrients.get("fiber_g") or 0.0
    sodium = nutrients.get("sodium_mg") or 0.0
    iron = nutrients.get("iron_mg") or 0.0
    potassium = nutrients.get("potassium_mg") or 0.0
    sugar = nutrients.get("sugar_g") or 0.0

    tags: list[str] = []
    _add_if(tags, "low-calorie", calories > 0 and calories <= 120)
    _add_if(tags, "high-protein", protein >= 10)
    _add_if(tags, "low-fat", fat <= 3)
    _add_if(tags, "low-carb", carbs <= 10)
    _add_if(tags, "high-fiber", fiber >= 5)
    _add_if(tags, "iron-rich", iron >= 2.5)
    _add_if(tags, "potassium-rich", potassium >= 400)
    _add_if(tags, "low-sodium", sodium <= 140)
    _add_if(tags, "diabetic-friendly", fiber >= 3 and sugar <= 5 and carbs <= 30)
    _add_if(tags, "heart-healthy", fat <= 5 and sodium <= 140 and fiber >= 3)
    _add_if(tags, "muscle-recovery", protein >= 10 and potassium >= 300)
    _add_if(tags, "keto-friendly", carbs <= 8 and fat >= 5)

    is_plant_based = _is_plant_based(food_name, category)
    _add_if(tags, "vegan", is_plant_based)
    _add_if(tags, "vegetarian", is_plant_based or not _has_animal_term(food_name))

    return tags


def generate_aliases(food_name: str, category: str) -> list[str]:
    normalized = food_name.lower()
    aliases = {food_name, normalized}

    simple = re.sub(r"\b(raw|cooked|boiled|roasted|commercial|canned)\b", "", normalized)
    simple = re.sub(r"\s+", " ", simple).strip(" ,")
    if simple:
        aliases.add(simple)

    for key, values in LEGUME_ALIASES.items():
        if key in normalized:
            aliases.update(values)

    if "beans" in normalized:
        aliases.add(normalized.replace("beans", "bean"))
    if "legume" in category.lower():
        aliases.add("legumes")

    return sorted(a for a in aliases if a)


def build_metadata(
    food: dict[str, Any],
    food_name: str,
    category: str,
    nutrients: dict[str, float | None],
    tags: list[str],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "food_id": str(food.get("fdcId", food.get("ndbNumber", ""))),
        "fdc_id": food.get("fdcId"),
        "ndb_number": food.get("ndbNumber"),
        "category": category,
        "data_type": food.get("dataType", ""),
        "food_name_normalized": food_name.lower(),
    }
    metadata.update({k: _round_or_none(v) for k, v in nutrients.items()})

    for tag in tags:
        metadata[tag.replace("-", "_")] = True

    for tag in (
        "high-protein", "low-fat", "low-carb", "high-fiber", "iron-rich",
        "potassium-rich", "low-sodium", "heart-healthy", "diabetic-friendly",
        "muscle-recovery", "vegan", "vegetarian", "keto-friendly",
    ):
        metadata.setdefault(tag.replace("-", "_"), False)

    return metadata


def build_semantic_text(
    food_name: str,
    category: str,
    nutrients: dict[str, float | None],
    tags: list[str],
    aliases: list[str],
) -> str:
    lines = [
        f"Food Name: {food_name}",
        f"Category: {category}",
        "",
        "Nutrition per 100g:",
    ]

    nutrient_labels = [
        ("calories", "Calories", "kcal"),
        ("protein_g", "Protein", "g"),
        ("fat_g", "Fat", "g"),
        ("carb_g", "Carbohydrates", "g"),
        ("fiber_g", "Fiber", "g"),
        ("sugar_g", "Sugar", "g"),
        ("sodium_mg", "Sodium", "mg"),
        ("iron_mg", "Iron", "mg"),
        ("potassium_mg", "Potassium", "mg"),
        ("calcium_mg", "Calcium", "mg"),
        ("magnesium_mg", "Magnesium", "mg"),
        ("zinc_mg", "Zinc", "mg"),
    ]
    for key, label, unit in nutrient_labels:
        value = nutrients.get(key)
        if value is not None:
            lines.append(f"- {label}: {_format_number(value)} {unit}")

    if tags:
        lines.extend(["", f"Tags: {', '.join(tags)}"])
        lines.append(_benefit_sentence(tags))
    if aliases:
        lines.append(f"Search aliases: {', '.join(aliases[:6])}.")

    return "\n".join(lines)


def write_json(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_app_compatible_json(path: Path, records: list[dict[str, Any]]) -> None:
    compatible = [to_app_record(record) for record in records]
    write_json(path, compatible)


def to_app_record(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record["metadata"]
    nutrients = []
    for field, label, unit in (
        ("protein_g", "protein", "g"),
        ("fat_g", "fat", "g"),
        ("carb_g", "carbohydrates", "g"),
        ("fiber_g", "fiber", "g"),
        ("sodium_mg", "sodium", "mg"),
        ("iron_mg", "iron", "mg"),
        ("potassium_mg", "potassium", "mg"),
        ("calcium_mg", "calcium", "mg"),
        ("magnesium_mg", "magnesium", "mg"),
        ("zinc_mg", "zinc", "mg"),
        ("sugar_g", "sugar", "g"),
    ):
        value = metadata.get(field)
        if value is not None:
            nutrients.append(f"{label}: {_format_number(value)}{unit}")

    return {
        "food_id": record["id"],
        "food_name": record["food_name"],
        "food_description": record["semantic_text"].split("\n\n", 1)[0],
        "food_ingredients": record["aliases"],
        "food_calories_per_serving": int(round(metadata.get("calories") or 0)),
        "food_nutritional_factors": "; ".join(nutrients),
        "food_health_benefits": _benefit_sentence(record["tags"]),
        "cooking_method": "",
        "cuisine_type": metadata.get("category", "Unknown"),
        "taste_profile": ", ".join(record["tags"]),
        "dietary_tags": record["tags"],
        "allergens": [],
        "semantic_text": record["semantic_text"],
        "metadata": metadata,
        "aliases": record["aliases"],
        "tags": record["tags"],
    }


def normalize_food_name(name: str) -> str:
    name = _clean_text(name)
    name = re.sub(r"\([^)]*% moisture[^)]*\)", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s+", " ", name).strip(" ,")
    if name.isupper():
        name = name.title()
    return name


def _clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_available(raw_nutrients: dict[str, list[float]], aliases: tuple[str, ...]) -> float | None:
    for alias in aliases:
        values = raw_nutrients.get(alias)
        if values:
            return round(values[0], 3)
    return None


def _round_or_none(value: float | None) -> float | None:
    return round(value, 3) if value is not None else None


def _format_number(value: float) -> str:
    if abs(value - round(value)) < 0.05:
        return str(int(round(value)))
    return f"{value:.1f}".rstrip("0").rstrip(".")


def _add_if(tags: list[str], tag: str, condition: bool) -> None:
    if condition and tag not in tags:
        tags.append(tag)


def _has_animal_term(food_name: str) -> bool:
    tokens = set(re.findall(r"[a-z]+", food_name.lower()))
    return bool(tokens & ANIMAL_TERMS)


def _is_plant_based(food_name: str, category: str) -> bool:
    category_l = category.lower()
    name_l = food_name.lower()
    plant_categories = ("legume", "vegetable", "fruit", "nut", "seed", "grain", "cereal")
    return any(c in category_l for c in plant_categories) and not _has_animal_term(name_l)


def _benefit_sentence(tags: list[str]) -> str:
    if not tags:
        return "This food has a measured USDA nutrient profile for nutrition-aware search."
    readable = ", ".join(tag.replace("-", " ") for tag in tags[:5])
    return f"This food is suitable for {readable} nutrition searches and recommendation filters."


def _dedupe_key(
    food_name: str,
    category: str,
    nutrients: dict[str, float | None],
) -> str:
    nutrient_bits = [
        str(round(nutrients.get(key) or 0, 1))
        for key in ("calories", "protein_g", "fat_g", "carb_g", "fiber_g")
    ]
    normalized_name = re.sub(r"[^a-z0-9]+", " ", food_name.lower()).strip()
    return "|".join([normalized_name, category.lower(), *nutrient_bits])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess USDA Foundation Foods for Food RAG.")
    parser.add_argument("--input", required=True, help="Path to USDA FoodData Central foundation JSON.")
    parser.add_argument("--output", default="data/usda_foundation_food_rag.json")
    parser.add_argument("--jsonl-output", default="data/usda_foundation_food_rag.jsonl")
    parser.add_argument("--app-output", default="data/FoodDataSet.usda.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = preprocess_usda_dataset(Path(args.input))
    write_json(Path(args.output), records)
    write_jsonl(Path(args.jsonl_output), records)
    write_app_compatible_json(Path(args.app_output), records)
    print(f"Processed {len(records)} foods")
    print(f"RAG JSON: {args.output}")
    print(f"RAG JSONL: {args.jsonl_output}")
    print(f"App-compatible JSON: {args.app_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
