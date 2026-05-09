from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


@dataclass
class Settings:
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    ollama_model_name: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.2:3b"))
    ollama_timeout_seconds: int = field(default_factory=lambda: int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120")))
    ollama_max_retries: int = field(default_factory=lambda: int(os.getenv("OLLAMA_MAX_RETRIES", "2")))
    embedding_model_name: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    chroma_db_path: str = field(default_factory=lambda: os.getenv("CHROMA_DB_PATH", "./chroma_db"))
    collection_name: str = field(default_factory=lambda: os.getenv("COLLECTION_NAME", "food_collection"))
    dataset_path: str = field(default_factory=lambda: os.getenv("DATASET_PATH", "./data/FoodDataSet.json"))
    default_n_results: int = field(default_factory=lambda: int(os.getenv("DEFAULT_N_RESULTS", "5")))
    retrieval_multiplier: int = field(default_factory=lambda: int(os.getenv("RETRIEVAL_MULTIPLIER", "10")))
    confidence_threshold: float = field(default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD", "0.0")))
    max_output_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_OUTPUT_TOKENS", "1024")))
    temperature: float = field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0.5")))
    cache_ttl_seconds: int = field(default_factory=lambda: int(os.getenv("CACHE_TTL_SECONDS", "3600")))
    batch_size: int = field(default_factory=lambda: int(os.getenv("BATCH_SIZE", "100")))
    host: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "8000")))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def clear_settings_cache() -> None:
    get_settings.cache_clear()
