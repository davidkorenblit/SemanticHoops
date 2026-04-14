"""
db/indexer.py

High-level abstraction for upserting and searching vectors in Qdrant.

Responsible for:
  - Batching upserts efficiently (avoids per-frame round-trips)
  - Cosine similarity search with optional metadata filtering
  - Returning typed SearchResult objects consumed by the API layer
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger
from qdrant_client.http.models import PointStruct, Filter

from .qdrant_client import QdrantClientWrapper


@dataclass
class SearchResult:
    """A single result returned by the vector similarity search."""
    id: str
    score: float
    payload: Dict[str, Any] = field(default_factory=dict)


class VectorIndexer:
    """
    Upserts frame embeddings into Qdrant and runs similarity searches.

    Args:
        client:          A connected QdrantClientWrapper instance.
        collection_name: Target Qdrant collection.
        batch_size:      Number of vectors per upsert batch.
    """

    def __init__(
        self,
        client: QdrantClientWrapper,
        collection_name: str,
        batch_size: int = 64,
    ) -> None:
        self._client = client.raw
        self.collection_name = collection_name
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def upsert(
        self,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> int:
        """
        Upsert a batch of vectors with associated metadata payloads.

        Args:
            vectors:  List of float32 embedding vectors.
            payloads: List of metadata dicts (same length as vectors).
            ids:      Optional list of UUIDs; auto-generated if omitted.

        Returns:
            Number of vectors successfully upserted.
        """
        if len(vectors) != len(payloads):
            raise ValueError("vectors and payloads must have the same length.")

        ids = ids or [str(uuid.uuid4()) for _ in vectors]

        total = 0
        for i in range(0, len(vectors), self.batch_size):
            batch_points = [
                PointStruct(id=ids[j], vector=vectors[j], payload=payloads[j])
                for j in range(i, min(i + self.batch_size, len(vectors)))
            ]
            self._client.upsert(
                collection_name=self.collection_name,
                points=batch_points,
            )
            total += len(batch_points)
            logger.debug(f"Upserted batch {i // self.batch_size + 1} ({total} total)")

        logger.success(f"Upserted {total} vectors → '{self.collection_name}'")
        return total

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        query_filter: Optional[Filter] = None,
    ) -> List[SearchResult]:
        """
        Run a cosine similarity search.

        Args:
            query_vector:  The query embedding (same dimensionality as indexed).
            top_k:         Number of results to return.
            query_filter:  Optional Qdrant Filter for metadata-based filtering.

        Returns:
            List of SearchResult ordered by descending similarity score.
        """
        hits = self._client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        return [
            SearchResult(id=str(h.id), score=h.score, payload=h.payload or {})
            for h in hits
        ]
