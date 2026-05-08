from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class TTLCache:
    def __init__(self, ttl_seconds: int = 3600) -> None:
        self.ttl_seconds = ttl_seconds
        self._items: dict[str, tuple[str, float]] = {}

    def get(self, key: str) -> str | None:
        item = self._items.get(key)
        if item is None:
            return None
        value, timestamp = item
        if time.time() - timestamp > self.ttl_seconds:
            del self._items[key]
            return None
        return value

    def set(self, key: str, value: str) -> None:
        self._items[key] = (value, time.time())

    def clear(self) -> None:
        self._items.clear()

    def size(self) -> int:
        return len(self._items)


class SemanticCache:
    def __init__(
        self,
        embedding_fn: Any,
        threshold: float = 0.92,
        ttl_seconds: int = 3600,
        max_entries: int = 500,
    ) -> None:
        self._embedding_fn = embedding_fn
        self._threshold = threshold
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._entries: list[tuple[list[float], str, float]] = []

    def get(self, query: str) -> str | None:
        try:
            q_vec = self._embed(query)
            if q_vec is None:
                return None

            now = time.time()
            best_score = -1.0
            best_response: str | None = None

            for vec, response, ts in self._entries:
                if now - ts > self._ttl:
                    continue
                score = _cosine(q_vec, vec)
                if score > best_score:
                    best_score = score
                    best_response = response

            if best_score >= self._threshold and best_response is not None:
                logger.debug("Semantic cache HIT (score=%.3f)", best_score)
                return best_response
        except Exception as exc:
            logger.warning("Semantic cache lookup error: %s", exc)
        return None

    def set(self, query: str, response: str) -> None:
        try:
            vec = self._embed(query)
            if vec is None:
                return
            self._entries.append((vec, response, time.time()))
            if len(self._entries) > self._max_entries:
                self._entries = self._entries[-self._max_entries:]
        except Exception as exc:
            logger.warning("Semantic cache write error: %s", exc)

    def _embed(self, text: str) -> list[float] | None:
        try:
            result = self._embedding_fn([text])
            if hasattr(result, "tolist"):
                return result[0].tolist()
            return list(result[0])
        except Exception as exc:
            logger.warning("Semantic cache embed error: %s", exc)
            return None

    def clear(self) -> None:
        self._entries.clear()

    def size(self) -> int:
        return len(self._entries)


class HybridCache:
    def __init__(
        self,
        ttl_seconds: int = 3600,
        embedding_fn: Any = None,
        semantic_threshold: float = 0.92,
    ) -> None:
        self._exact = TTLCache(ttl_seconds=ttl_seconds)
        self._semantic: SemanticCache | None = (
            SemanticCache(
                embedding_fn=embedding_fn,
                threshold=semantic_threshold,
                ttl_seconds=ttl_seconds,
            )
            if embedding_fn is not None
            else None
        )

    def get(self, key: str, query: str = "") -> str | None:
        hit = self._exact.get(key)
        if hit is not None:
            logger.debug("Exact cache HIT")
            return hit

        if self._semantic and query:
            hit = self._semantic.get(query)
            if hit is not None:
                return hit

        return None

    def set(self, key: str, query: str, value: str) -> None:
        self._exact.set(key, value)
        if self._semantic and query:
            self._semantic.set(query, value)

    def clear(self) -> None:
        self._exact.clear()
        if self._semantic:
            self._semantic.clear()

    def stats(self) -> dict[str, int]:
        return {
            "exact_entries": self._exact.size(),
            "semantic_entries": self._semantic.size() if self._semantic else 0,
        }


def make_cache_key(query: str, n_results: int, filters: Any) -> str:
    if dataclasses.is_dataclass(filters):
        filter_payload = dataclasses.asdict(filters)
    else:
        filter_payload = filters
    payload = {
        "query": query.strip().lower(),
        "n_results": n_results,
        "filters": filter_payload,
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cosine(a: list[float], b: list[float]) -> float:
    try:
        import numpy as np
        va, vb = np.array(a), np.array(b)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        return float(np.dot(va, vb) / denom) if denom > 0 else 0.0
    except Exception:
        return 0.0
