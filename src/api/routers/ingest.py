"""
api/routers/ingest.py

POST /ingest — trigger the full ingestion pipeline for a single video URL.

Flow:
  1. Download video via yt-dlp (VideoDownloader).
  2. Extract frames via Smart Sampling (SmartSampler).
  3. Embed each frame via CLIP (CLIPWrapper).
  4. Upsert embeddings + metadata into Qdrant (VectorIndexer).

NOTE: In production this should be dispatched to a background task queue
(e.g. Celery + Redis). The current implementation runs synchronously and is
suitable for the Phase 1-2 prototype.
"""
from __future__ import annotations

import os
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks

from ..schemas import IngestRequest, IngestResponse
from ...ingestion.downloader import VideoDownloader
from ...ingestion.sampler import SmartSampler
from ...db.indexer import VectorIndexer

router = APIRouter()


def _run_pipeline(
    url: str,
    sport: str | None,
    description: str | None,
    embedder,
    qdrant_client,
) -> IngestResponse:
    """Synchronous pipeline: download → sample → embed → index."""

    # Step 1: Download
    downloader = VideoDownloader()
    video_path = downloader.download(url)

    # Step 2: Smart Sample
    sampler = SmartSampler(
        motion_threshold=float(os.getenv("MOTION_THRESHOLD", "0.02")),
        max_fps=float(os.getenv("SMART_SAMPLE_FPS_MAX", "10")),
    )
    frames = sampler.sample(video_path)

    if not frames:
        raise ValueError("No frames extracted from video.")

    # Step 3: Embed
    vectors = [embedder.embed_image(f.image).tolist() for f in frames]
    payloads = [
        {
            "video_id": f.video_id,
            "timestamp_sec": f.timestamp_sec,
            "frame_index": f.frame_index,
            "source_url": url,
            "sport": sport,
            "description": description,
        }
        for f in frames
    ]

    # Step 4: Index
    collection = os.getenv("QDRANT_COLLECTION_NAME", "sport_clips")
    indexer = VectorIndexer(client=qdrant_client, collection_name=collection)
    indexed = indexer.upsert(vectors=vectors, payloads=payloads)

    return IngestResponse(
        status="success",
        video_id=frames[0].video_id,
        frames_indexed=indexed,
        message=f"Indexed {indexed} frames from '{frames[0].video_id}'.",
    )


@router.post(
    "/",
    response_model=IngestResponse,
    summary="Download and index a video URL",
)
async def ingest_video(
    body: IngestRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> IngestResponse:
    try:
        return _run_pipeline(
            url=body.url,
            sport=body.sport,
            description=body.description,
            embedder=request.app.state.embedder,
            qdrant_client=request.app.state.qdrant,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
