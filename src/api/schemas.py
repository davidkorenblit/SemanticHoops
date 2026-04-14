"""
api/schemas.py

Pydantic v2 request / response models for the Sport-Tech search API.

Strict validation is intentional — callers get clear 422 errors instead of
silent bad data propagating into the vector DB or ML pipeline.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Request Models
# =============================================================================

class SearchRequest(BaseModel):
    """Payload for /search endpoint — text-to-video semantic search."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Natural-language search query (e.g. 'basketball slam dunk').",
        examples=["basketball slam dunk courtside"],
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of results to return.",
    )
    sport_filter: Optional[str] = Field(
        default=None,
        description="Optional sport category filter (e.g. 'basketball', 'football').",
    )

    @field_validator("query")
    @classmethod
    def strip_and_validate(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query must not be empty or whitespace.")
        return v


class IngestRequest(BaseModel):
    """Payload for /ingest endpoint — trigger video download + indexing."""

    url: str = Field(
        ...,
        description="YouTube (or yt-dlp-compatible) URL to download and index.",
    )
    sport: Optional[str] = Field(
        default=None,
        description="Sport category tag stored in the vector payload.",
    )
    description: Optional[str] = Field(
        default=None,
        max_length=1024,
        description="Optional human-readable description of the clip.",
    )


# =============================================================================
# Response Models
# =============================================================================

class ClipResult(BaseModel):
    """A single search result corresponding to a video clip / frame."""

    id: str = Field(description="Unique vector ID in Qdrant.")
    score: float = Field(description="Cosine similarity score [0, 1].")
    video_id: str = Field(description="Source video identifier.")
    timestamp_sec: float = Field(description="Frame timestamp in the source video.")
    sport: Optional[str] = Field(default=None)
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Full metadata payload from Qdrant.",
    )


class SearchResponse(BaseModel):
    """Response envelope for /search."""

    query: str
    results: List[ClipResult]
    total: int = Field(description="Number of results returned.")


class IngestResponse(BaseModel):
    """Response envelope for /ingest."""

    status: str
    video_id: str
    frames_indexed: int
    message: str


class HealthResponse(BaseModel):
    """Response for /health liveness probe."""

    status: str          # "ok" | "degraded"
    qdrant_connected: bool
    model_loaded: bool
    version: str = "0.1.0"
