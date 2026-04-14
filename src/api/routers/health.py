"""
api/routers/health.py

GET /health — liveness and readiness probe for Docker and Kubernetes.

Returns 200 OK when both Qdrant is reachable and the CLIP model is loaded.
Returns 503 Service Unavailable if any dependency is unhealthy.
"""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..schemas import HealthResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness / Readiness probe",
)
async def health_check(request: Request) -> HealthResponse | JSONResponse:
    qdrant_ok = request.app.state.qdrant.health_check()
    model_ok = hasattr(request.app.state, "embedder")

    status = "ok" if (qdrant_ok and model_ok) else "degraded"
    response = HealthResponse(
        status=status,
        qdrant_connected=qdrant_ok,
        model_loaded=model_ok,
    )

    if status != "ok":
        return JSONResponse(status_code=503, content=response.model_dump())

    return response
