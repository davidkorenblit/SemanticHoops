import os
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from src.models.embeddings import CLIPWrapper

COLLECTION_NAME = "nba_frames"
DATA_DIR = Path("data/processed")

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

    print(f"Found {len(all_image_paths)} frames across all videos. Generating embeddings...")
    results = model.get_batch_images_embeddings(all_image_paths, batch_size=32)

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

    batch_size = 200
    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points[i:i + batch_size]
        )

    print(f"Upserted {len(points)} real frames to '{COLLECTION_NAME}'.")

if __name__ == "__main__":
    main()