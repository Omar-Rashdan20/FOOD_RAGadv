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
        self._index: Any | None = None
        self._index_entries: list[tuple[list[float], str, float]] = []
        self._dirty_index = True

    def get(self, query: str) -> str | None:
        try:
            q_vec = self._embed(query)
            if q_vec is None:
                return None
            q_vec = _normalise(q_vec)

            now = time.time()
            self._prune(now)
            best_score, best_response = self._nearest(q_vec)

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
            self._entries.append((_normalise(vec), response, time.time()))
            if len(self._entries) > self._max_entries:
                self._entries = self._entries[-self._max_entries:]
            self._dirty_index = True
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
        self._index = None
        self._index_entries = []
        self._dirty_index = True

    def size(self) -> int:
        return len(self._entries)

    def _prune(self, now: float) -> None:
        fresh = [entry for entry in self._entries if now - entry[2] <= self._ttl]
        if len(fresh) != len(self._entries):
            self._entries = fresh
            self._dirty_index = True

    def _nearest(self, q_vec: list[float]) -> tuple[float, str | None]:
        if not self._entries:
            return -1.0, None
        if self._build_index():
            distance, idx = self._index.query(q_vec, k=1)
            score = max(0.0, min(1.0, 1.0 - (float(distance) ** 2 / 2.0)))
            return score, self._index_entries[int(idx)][1]

        best_score = -1.0
        best_response: str | None = None
        for vec, response, _ts in self._entries:
            score = _cosine(q_vec, vec)
            if score > best_score:
                best_score = score
                best_response = response
        return best_score, best_response

    def _build_index(self) -> bool:
        if not self._dirty_index and self._index is not None:
            return True
        try:
            from scipy.spatial import KDTree
            self._index_entries = list(self._entries)
            vectors = [entry[0] for entry in self._index_entries]
            self._index = KDTree(vectors)
            self._dirty_index = False
            return True
        except Exception as exc:
            logger.debug("Semantic cache KDTree disabled: %s", exc)
            self._index = None
            self._index_entries = []
            self._dirty_index = False
            return False


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
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        denom = norm_a * norm_b
        return dot / denom if denom > 0 else 0.0


def _normalise(vec: list[float]) -> list[float]:
    try:
        import numpy as np
        arr = np.array(vec, dtype=float)
        norm = np.linalg.norm(arr)
        if norm <= 0:
            return list(arr)
        return list(arr / norm)
    except Exception:
        norm = sum(x * x for x in vec) ** 0.5
        return [x / norm for x in vec] if norm > 0 else vec
