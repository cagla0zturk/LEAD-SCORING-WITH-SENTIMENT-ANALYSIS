"""FastAPI application exposing the lead-priority service.

Endpoints
---------
* ``GET  /health``      - liveness + whether models are loaded.
* ``POST /score``       - score a single lead (conversion + sentiment + priority).
* ``GET  /leads/top``   - rank a demo lead list and return the top N by priority.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Query, Request

from lead_priority.api.logging_config import configure_logging
from lead_priority.api.schemas import (
    HealthResponse,
    MorningBriefResponse,
    ScoreRequest,
    ScoreResponse,
    TopLeadsResponse,
)
from lead_priority.api.service import PriorityService

configure_logging()
logger = logging.getLogger("lead_priority.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load models once at startup; degrade gracefully if artifacts are missing."""
    try:
        app.state.service = PriorityService.load()
        logger.info("Models loaded successfully at startup.")
    except FileNotFoundError as exc:
        app.state.service = None
        logger.warning("Models not loaded: %s. Train with `python -m scripts.train_all`.", exc)
    yield


app = FastAPI(
    title="Lead Priority API",
    description=(
        "Lead scoring + engagement sentiment + combined priority score for a CRM "
        "sales workflow. See README for the modelling and leakage decisions."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request with its latency (basic observability)."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s -> %d (%.1f ms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


def _get_service(request: Request) -> PriorityService:
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Models are not loaded. Train them with `python -m scripts.train_all`.",
        )
    return service


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    service = getattr(request.app.state, "service", None)
    return HealthResponse(
        status="ok",
        models_loaded=service is not None,
        n_demo_leads=len(service.demo_leads) if service else 0,
    )


@app.post("/score", response_model=ScoreResponse)
def score(payload: ScoreRequest, request: Request) -> ScoreResponse:
    service = _get_service(request)
    result = service.score_lead(
        features=payload.features,
        interaction_text=payload.interaction_text,
        conversion_weight=payload.conversion_weight,
    )
    return ScoreResponse(**result)


@app.get("/leads/top", response_model=TopLeadsResponse)
def top_leads(
    request: Request,
    n: int = Query(5, ge=1, le=100, description="Number of top leads to return."),
    conversion_weight: float | None = Query(
        None, ge=0.0, le=1.0, description="Override the conversion-vs-sentiment weight."
    ),
) -> TopLeadsResponse:
    service = _get_service(request)
    leads = service.top_leads(n=n, conversion_weight=conversion_weight)
    return TopLeadsResponse(count=len(leads), leads=leads)


@app.get("/dashboard/brief", response_model=MorningBriefResponse)
def morning_brief(
    request: Request,
    n_call: int = Query(5, ge=1, le=100, description="How many leads to call today."),
    n_cooling: int = Query(5, ge=1, le=100, description="How many cooling/at-risk leads."),
    conversion_weight: float | None = Query(
        None, ge=0.0, le=1.0, description="Override the conversion-vs-sentiment weight."
    ),
) -> MorningBriefResponse:
    """The sales-rep morning brief: 'call these N today' + 'these are cooling off'."""
    service = _get_service(request)
    brief = service.morning_brief(
        n_call=n_call, n_cooling=n_cooling, conversion_weight=conversion_weight
    )
    return MorningBriefResponse(**brief)
