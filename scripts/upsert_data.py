import os
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

COLLECTION_NAME = "nba_frames"

DUMMY_FRAMES = [
    {"id": 1, "video_id": "lakers_vs_celtics_2024", "timestamp": 12, "image_path": "data/processed/frames/lakers_vs_celtics_2024/frame_0012.jpg"},
    {"id": 2, "video_id": "lakers_vs_celtics_2024", "timestamp": 45, "image_path": "data/processed/frames/lakers_vs_celtics_2024/frame_0045.jpg"},
    {"id": 3, "video_id": "lakers_vs_celtics_2024", "timestamp": 78, "image_path": "data/processed/frames/lakers_vs_celtics_2024/frame_0078.jpg"},
    {"id": 4, "video_id": "warriors_vs_bucks_2024", "timestamp": 23, "image_path": "data/processed/frames/warriors_vs_bucks_2024/frame_0023.jpg"},
    {"id": 5, "video_id": "warriors_vs_bucks_2024", "timestamp": 56, "image_path": "data/processed/frames/warriors_vs_bucks_2024/frame_0056.jpg"},
]

def main():
    client = QdrantClient(
        host=os.getenv("QDRANT_HOST", "qdrant"),
        port=int(os.getenv("QDRANT_PORT", 6333)),
    )

    np.random.seed(42)

    points = []
    for frame in DUMMY_FRAMES:
        vector = np.random.randn(512).astype(np.float32).tolist()
        points.append(
            PointStruct(
                id=frame["id"],
                vector=vector,
                payload={
                    "video_id": frame["video_id"],
                    "timestamp": frame["timestamp"],
                    "image_path": frame["image_path"],
                },
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Upserted {len(points)} points to '{COLLECTION_NAME}'.")

if __name__ == "__main__":
    main()
