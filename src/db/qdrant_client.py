"""
db/qdrant_client.py

Thin wrapper around qdrant-client that:
  - Reads connection settings from environment / .env
  - Re-exposes only the methods actually used by this project
  - Makes the client swappable in tests via dependency injection
"""
from __future__ import annotations

import os
from typing import List

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    CollectionStatus,
)


class QdrantClientWrapper:
    """
    Manages the connection to a Qdrant instance.

    Args:
        host:       Qdrant hostname (default: QDRANT_HOST env var or "localhost").
        port:       Qdrant REST port (default: QDRANT_PORT env var or 6333).
        api_key:    Optional API key for Qdrant Cloud.
        prefer_grpc: Use gRPC for upserts (faster for bulk operations).
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        api_key: str | None = None,
        prefer_grpc: bool = True,
    ) -> None:
        self.host = host or os.getenv("QDRANT_HOST", "localhost")
        self.port = port or int(os.getenv("QDRANT_PORT", "6333"))
        self.api_key = api_key or os.getenv("QDRANT_API_KEY") or None

        self._client = QdrantClient(
            host=self.host,
            port=self.port,
            api_key=self.api_key,
            prefer_grpc=prefer_grpc,
        )
        logger.info(f"Qdrant client → {self.host}:{self.port}")

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """Return True if Qdrant is reachable and healthy."""
        try:
            self._client.get_collections()
            return True
        except Exception as exc:
            logger.error(f"Qdrant health check failed: {exc}")
            return False

    def ensure_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: Distance = Distance.COSINE,
    ) -> None:
        """Create the collection if it does not already exist."""
        existing = {c.name for c in self._client.get_collections().collections}
        if collection_name not in existing:
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance),
            )
            logger.success(f"Created collection '{collection_name}' (dim={vector_size})")
        else:
            logger.debug(f"Collection '{collection_name}' already exists.")

    # ------------------------------------------------------------------
    # Raw client access (escape hatch for advanced usage)
    # ------------------------------------------------------------------

    @property
    def raw(self) -> QdrantClient:
        """Access the underlying qdrant-client instance directly."""
        return self._client
