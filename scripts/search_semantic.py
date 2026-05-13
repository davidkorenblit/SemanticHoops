import os
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path
from qdrant_client import QdrantClient
from src.models.embeddings import CLIPWrapper

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    client = QdrantClient(
        host=os.getenv("QDRANT_HOST", "qdrant"),
        port=int(os.getenv("QDRANT_PORT", 6333))
    )

    model = CLIPWrapper()
    query_vector = model.get_text_embedding(args.query)

    if hasattr(query_vector, "tolist"):
        query_vector = query_vector.tolist()

    search_result = client.search(
        collection_name="nba_frames",
        query_vector=query_vector,
        limit=args.top_k,
        with_payload=True
    )

    results_data = []
    for hit in search_result:
        score = hit.score
        payload = hit.payload
        video_id = payload.get("video_id")
        timestamp = payload.get("timestamp")
        image_path = payload.get("image_path")

        print(f"Score: {score:.4f} | Video ID: {video_id} | Timestamp: {timestamp} | Path: {image_path}")

        results_data.append({
            "score": score,
            "payload": payload
        })

    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "query": args.query,
        "search_timestamp": datetime.now(timezone.utc).isoformat(),
        "top_matches": results_data
    }

    with open(log_dir / "search_results.json", "w") as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()
