import os
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path
from qdrant_client import QdrantClient
from src.models.embeddings import CLIPWrapper
from src.models.search import aggregate_temporal_windows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=5, help="Number of final results to return")
    parser.add_argument("--raw_k", type=int, default=50, help="Number of raw frame candidates to pull from vector DB when pooling is enabled")
    parser.add_argument("--window", type=float, default=10.0, help="Sliding window size in seconds")
    parser.add_argument("--no_pooling", action="store_true", help="Disable temporal pooling and return raw frames instead")
    args = parser.parse_args()

    client = QdrantClient(
        host=os.getenv("QDRANT_HOST", "qdrant"),
        port=int(os.getenv("QDRANT_PORT", 6333))
    )

    model = CLIPWrapper()
    query_vector = model.get_text_embedding(args.query)

    if hasattr(query_vector, "tolist"):
        query_vector = query_vector.tolist()

    # Determine limit based on whether pooling is enabled
    limit = args.top_k if args.no_pooling else args.raw_k

    search_result = client.search(
        collection_name="nba_frames",
        query_vector=query_vector,
        limit=limit,
        with_payload=True
    )

    results_data = []

    if args.no_pooling:
        print("\n=== SEMANTIC SEARCH RESULTS (RAW FRAMES) ===")
        for hit in search_result:
            score = hit.score
            payload = hit.payload
            video_id = payload.get("video_id")
            timestamp = payload.get("timestamp")
            image_path = payload.get("image_path")

            print(f"Score: {score:.4f} | Video ID: {video_id} | Timestamp: {timestamp}s | Path: {image_path}")

            results_data.append({
                "score": score,
                "payload": payload
            })
    else:
        print(f"\n=== SEMANTIC SEARCH RESULTS (TEMPORAL WINDOWS, size={args.window}s) ===")
        windows = aggregate_temporal_windows(search_result, window_seconds=args.window)
        
        # Display top_k windows
        for idx, w in enumerate(windows[:args.top_k], 1):
            print(
                f"Rank {idx} | Window Score: {w['window_score']:.4f} (Peak: {w['peak_score']:.4f}) | "
                f"Video ID: {w['video_id']} | Interval: {w['window_start']:.1f}s - {w['window_end']:.1f}s | "
                f"Representative Frame: {w['representative_frame']} | Hit Count: {w['hit_count']}"
            )
            
            # Formulate result dictionary matching format
            results_data.append({
                "window_score": w["window_score"],
                "peak_score": w["peak_score"],
                "video_id": w["video_id"],
                "window_start": w["window_start"],
                "window_end": w["window_end"],
                "representative_frame": w["representative_frame"],
                "hit_count": w["hit_count"],
                "hits_in_window": [
                    {"score": h.score, "payload": h.payload} if hasattr(h, "score") else h
                    for h in w["hits"]
                ]
            })

    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "query": args.query,
        "search_timestamp": datetime.now(timezone.utc).isoformat(),
        "pooling_enabled": not args.no_pooling,
        "window_size_seconds": args.window if not args.no_pooling else None,
        "results": results_data
    }

    with open(log_dir / "search_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults logged to {log_dir / 'search_results.json'}")

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
