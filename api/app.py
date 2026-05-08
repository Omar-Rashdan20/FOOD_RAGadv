from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from src.config import get_settings
from src.rag_pipeline import FoodRAGPipeline, build_pipeline

logger = logging.getLogger(__name__)

_pipeline: FoodRAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    settings = get_settings()
    logger.info("Building RAG pipeline...")
    _pipeline = build_pipeline(settings=settings)
    logger.info("Pipeline ready.")
    yield
    _pipeline = None


app = FastAPI(
    title="Food RAG API",
    description="Production-grade food recommendation API powered by hybrid RAG.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_latency_header(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    response.headers["X-Response-Time-Ms"] = str(round((time.monotonic() - start) * 1000))
    return response


def _get_pipeline() -> FoodRAGPipeline:
    if _pipeline is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Pipeline not ready.")
    return _pipeline


from api.routes import eval_router, health_router, recommend_router
app.include_router(health_router)
app.include_router(recommend_router)
app.include_router(eval_router)
