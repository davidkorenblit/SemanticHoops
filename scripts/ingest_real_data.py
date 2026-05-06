import os
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from src.models.embeddings import CLIPWrapper

COLLECTION_NAME = "nba_frames"
DATA_DIR = Path("data/processed/vid1")

def main():
    client = QdrantClient(
        host=os.getenv("QDRANT_HOST", "qdrant"),
        port=int(os.getenv("QDRANT_PORT", 6333)),
    )

    model = CLIPWrapper()

    image_paths = sorted(DATA_DIR.glob("*.jpg"))
    if not image_paths:
        print(f"No .jpg files found in {DATA_DIR}")
        return

    print(f"Found {len(image_paths)} frames. Generating embeddings...")
    results = model.get_batch_images_embeddings(image_paths, batch_size=32)

    points = []
    for i, item in enumerate(results):
        points.append(
            PointStruct(
                id=i + 1,
                vector=item["embedding"],
                payload={
                    "video_id": item["video_id"],
                    "timestamp": item["timestamp"],
                    "image_path": item["file_path"],
                },
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Upserted {len(points)} real frames to '{COLLECTION_NAME}'.")

if __name__ == "__main__":
    main()
