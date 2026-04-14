"""
api/routers/search.py

POST /search — text-to-video semantic search.

Flow:
  1. Validate + normalise query string (schema layer).
  2. Encode query via CLIP (models layer).
  3. Run cosine similarity search (db layer).
  4. Map Qdrant hits → ClipResult response objects.
"""
from __future__ import annotations

from fastapi import APIRouter, Request, HTTPException

from ..schemas import SearchRequest, SearchResponse, ClipResult
from ...db.indexer import VectorIndexer
import os

router = APIRouter()


@router.post(
    "/",
    response_model=SearchResponse,
    summary="Semantic text-to-video search",
)
async def semantic_search(body: SearchRequest, request: Request) -> SearchResponse:
    embedder = request.app.state.embedder
    qdrant_client = request.app.state.qdrant

    # 1. Encode query
    try:
        query_vector = embedder.embed_text(body.query).tolist()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {exc}")

    # 2. Build optional metadata filter
    qdrant_filter = None
    if body.sport_filter:
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue
        qdrant_filter = Filter(
            must=[FieldCondition(key="sport", match=MatchValue(value=body.sport_filter))]
        )

    # 3. Search
    collection = os.getenv("QDRANT_COLLECTION_NAME", "sport_clips")
    indexer = VectorIndexer(client=qdrant_client, collection_name=collection)
    hits = indexer.search(
        query_vector=query_vector,
        top_k=body.top_k,
        query_filter=qdrant_filter,
    )

    # 4. Map to response schema
    results = [
        ClipResult(
            id=h.id,
            score=h.score,
            video_id=h.payload.get("video_id", "unknown"),
            timestamp_sec=h.payload.get("timestamp_sec", 0.0),
            sport=h.payload.get("sport"),
            payload=h.payload,
        )
        for h in hits
    ]

    return SearchResponse(query=body.query, results=results, total=len(results))
