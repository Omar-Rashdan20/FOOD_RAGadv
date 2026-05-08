from __future__ import annotations

from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, examples=["healthy Indian food under 400 calories, no nuts"])
    n_results: int = Field(default=5, ge=1, le=20)
    use_cache: bool = Field(default=True)


class RecommendResponse(BaseModel):
    query: str
    answer: str
    cached: bool = False
    latency_ms: float


class CacheStatsResponse(BaseModel):
    exact_entries: int
    semantic_entries: int


class HealthResponse(BaseModel):
    status: str
    pipeline_ready: bool
    cache: CacheStatsResponse


class EvalSampleRequest(BaseModel):
    query: str
    ground_truth_answer: str = ""
    relevant_doc_ids: list[str] = Field(default_factory=list)


class EvalReportResponse(BaseModel):
    n_samples: int
    retrieval: dict
    generation: dict
    passed_targets: list[str]
    failed_targets: list[str]
