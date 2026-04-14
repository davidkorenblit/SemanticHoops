"""
tests/test_health.py

Smoke test: ensure /health returns 200 when dependencies are mocked.
Uses FastAPI's TestClient (ASGI) and monkeypatching — no real GPU or Qdrant needed.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from src.api.main import create_app


@pytest.fixture()
def client():
    """TestClient with mocked app state (no real CLIP or Qdrant)."""
    app = create_app()

    # Patch lifespan so we don't actually connect to Qdrant or load CLIP
    mock_qdrant = MagicMock()
    mock_qdrant.health_check.return_value = True
    mock_embedder = MagicMock()
    mock_embedder.model_name = "MockCLIP"
    mock_embedder.embedding_dim = 512

    app.state.qdrant = mock_qdrant
    app.state.embedder = mock_embedder

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["qdrant_connected"] is True
    assert data["model_loaded"] is True
