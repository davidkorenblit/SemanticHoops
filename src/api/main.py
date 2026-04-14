"""
api/main.py

FastAPI application factory and router registration.

Lifespan:
  - On startup: initialise CLIP model, Qdrant client, and ensure collection.
  - On shutdown: release GPU memory and close DB connections.

Endpoints (see individual routers for full docstrings):
  GET  /health      — liveness / readiness probe
  POST /search      — text-to-video semantic search
  POST /ingest      — trigger video download + indexing pipeline
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .routers import health, search, ingest
from ..db.qdrant_client import QdrantClientWrapper
from ..models.clip_wrapper import CLIPWrapper


# =============================================================================
# Application State (shared across requests via app.state)
# =============================================================================

class AppState:
    qdrant: QdrantClientWrapper
    embedder: CLIPWrapper


# =============================================================================
# Lifespan — replaces @app.on_event (deprecated in FastAPI ≥ 0.93)
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # ── Startup ──────────────────────────────────────────────────────────────
    logger.info("Initialising application resources...")

    model_name = os.getenv("CLIP_MODEL_NAME", "ViT-B/32")
    device = os.getenv("CLIP_DEVICE", "cpu")

    app.state.embedder = CLIPWrapper(model_name=model_name, device=device)
    logger.success(f"CLIP model loaded: {app.state.embedder.model_name}")

    app.state.qdrant = QdrantClientWrapper()
    app.state.qdrant.ensure_collection(
        collection_name=os.getenv("QDRANT_COLLECTION_NAME", "sport_clips"),
        vector_size=app.state.embedder.embedding_dim,
    )
    logger.success("Qdrant ready.")

    yield  # ← application runs here

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("Shutting down — releasing resources...")
    # PyTorch / CUDA cleanup happens via GC; explicit del for clarity.
    del app.state.embedder


# =============================================================================
# Application Factory
# =============================================================================

def create_app() -> FastAPI:
    app = FastAPI(
        title="Semantic Sport-Tech API",
        description=(
            "Text-to-video semantic search over sports footage. "
            "Powered by CLIP embeddings and Qdrant vector DB."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — tighten allowed_origins before going to production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(search.router, prefix="/search", tags=["Search"])
    app.include_router(ingest.router, prefix="/ingest", tags=["Ingest"])

    return app


# ASGI app instance (imported by main.py at the project root)
app = create_app()
