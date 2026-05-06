import os
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from src.models.embeddings import CLIPWrapper

COLLECTION_NAME = "nba_frames"
DATA_DIR = Path("data/processed")
LOG_DIR = Path("data/logs")
BATCH_SIZE = 32
UPSERT_BATCH_SIZE = 200

def main():
    client = QdrantClient(
        host=os.getenv("QDRANT_HOST", "qdrant"),
        port=int(os.getenv("QDRANT_PORT", 6333)),
    )

    model = CLIPWrapper()

    all_image_paths = sorted(DATA_DIR.rglob("*.jpg"))
    if not all_image_paths:
        print(f"No .jpg files found under {DATA_DIR}")
        return

    total_found = len(all_image_paths)
    print(f"Found {total_found} frames across all videos. Generating embeddings...")

    t_embed_start = time.perf_counter()
    results = model.get_batch_images_embeddings(all_image_paths, batch_size=BATCH_SIZE)
    t_embed_end = time.perf_counter()

    points = []
    for global_id, item in enumerate(results, start=1):
        points.append(
            PointStruct(
                id=global_id,
                vector=item["embedding"],
                payload={
                    "video_id": item["video_id"],
                    "timestamp": item["timestamp"],
                    "image_path": item["file_path"],
                },
            )
        )

    t_upsert_start = time.perf_counter()
    for i in range(0, len(points), UPSERT_BATCH_SIZE):
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points[i:i + UPSERT_BATCH_SIZE],
        )
    t_upsert_end = time.perf_counter()

    embedding_time = round(t_embed_end - t_embed_start, 2)
    upsert_time = round(t_upsert_end - t_upsert_start, 2)
    total_time = round(embedding_time + upsert_time, 2)

    print(f"Upserted {len(points)} real frames to '{COLLECTION_NAME}'.")
    print(f"Embedding: {embedding_time}s | Upsert: {upsert_time}s | Total: {total_time}s")

    report = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_frames_found": total_found,
        "total_frames_processed": len(points),
        "embedding_batch_size": BATCH_SIZE,
        "upsert_batch_size": UPSERT_BATCH_SIZE,
        "embedding_time_seconds": embedding_time,
        "upsert_time_seconds": upsert_time,
        "total_time_seconds": total_time,
    }

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    report_path = LOG_DIR / "ingestion_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()