from __future__ import annotations
import time
import logging
from fastapi import APIRouter, HTTPException, Request, status
from .schemas import (CacheStatsResponse,EvalReportResponse,EvalSampleRequest,HealthResponse,RecommendRequest,RecommendResponse,)
from src.evaluator import build_eval_samples, run_full_eval

logger = logging.getLogger(__name__)

health_router = APIRouter(tags=["health"])
recommend_router = APIRouter(prefix="/recommend", tags=["recommend"])
eval_router = APIRouter(prefix="/eval", tags=["eval"])


def _pipeline():
    from .app import _get_pipeline
    return _get_pipeline()


@health_router.get("/health", response_model=HealthResponse)
def health_check():
    try:
        p = _pipeline()
        stats = p.cache.stats()
        return HealthResponse(
            status="ok",
            pipeline_ready=True,
            cache=CacheStatsResponse(**stats),
        )
    except HTTPException:
        return HealthResponse(
            status="starting",
            pipeline_ready=False,
            cache=CacheStatsResponse(exact_entries=0, semantic_entries=0),
        )


@health_router.get("/", include_in_schema=False)
def root():
    return {"message": "Food RAG API — see /docs for usage."}


@recommend_router.post("", response_model=RecommendResponse)
def recommend(body: RecommendRequest, request: Request):
    p = _pipeline()
    start = time.monotonic()

    try:
        answer = p.rag_recommend(
            query=body.query,
            n_results=body.n_results,
            use_cache=body.use_cache,
            request_id=getattr(request.state, "request_id", None),
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        logger.exception("Recommendation failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    latency = round((time.monotonic() - start) * 1000, 2)
    return RecommendResponse(
        query=body.query,
        answer=answer,
        latency_ms=latency,
    )


@recommend_router.delete("/cache", status_code=status.HTTP_204_NO_CONTENT)
def clear_cache():
    _pipeline().clear_caches()


@eval_router.post("", response_model=EvalReportResponse)
def run_eval(samples: list[EvalSampleRequest]):
    if not samples:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="No samples provided.")

    p = _pipeline()


    eval_samples = build_eval_samples(p, samples, n_results=10)
    report = run_full_eval(eval_samples, llm_client=p.llm_client)
    return EvalReportResponse(**report.to_dict())
