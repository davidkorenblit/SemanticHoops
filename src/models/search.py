"""src/models/search.py
=================================================
Search Layer – Temporal Window Aggregation and Pooling.

* Implements algorithms to group individual frame search results into temporal windows.
* Includes sliding window pooling (average/peak scoring) and Non-Maximum Suppression (NMS)
  to deduplicate overlapping windows.
"""

from typing import List, Dict, Any, Union
import numpy as np

def aggregate_temporal_windows(
    hits: List[Any], 
    window_seconds: float = 10.0, 
    min_hits_in_window: int = 1
) -> List[Dict[str, Any]]:
    """
    Groups retrieved frame hits by video_id and aggregates their scores using a sliding window.
    Applies Non-Maximum Suppression (NMS) to remove overlapping windows.
    
    Args:
        hits: List of search results from Qdrant client. Each hit can be a Qdrant ScoredPoint 
              or a dict, and must have a score and payload containing 'video_id', 'timestamp', and 'image_path'.
        window_seconds: The duration of the sliding window in seconds.
        min_hits_in_window: Minimum number of frame hits required in a window to keep it.
        
    Returns:
        A sorted list of dictionaries representing the best non-overlapping windows:
        [
            {
                "video_id": str,
                "window_score": float (mean similarity score in window),
                "peak_score": float (highest score in window),
                "window_start": float,
                "window_end": float,
                "representative_frame": str,
                "hit_count": int,
                "hits": list of raw hits in this window
            },
            ...
        ]
    """
    # 1. Group hits by video_id
    video_groups: Dict[str, List[Dict[str, Any]]] = {}
    for hit in hits:
        if hasattr(hit, "payload") and hit.payload:
            payload = hit.payload
            score = hit.score
        elif isinstance(hit, dict):
            payload = hit.get("payload", {})
            score = hit.get("score", 0.0)
        else:
            continue
            
        video_id = payload.get("video_id")
        timestamp = payload.get("timestamp")
        image_path = payload.get("image_path")
        
        if not video_id or timestamp is None:
            continue
            
        if video_id not in video_groups:
            video_groups[video_id] = []
            
        video_groups[video_id].append({
            "score": float(score),
            "timestamp": float(timestamp),
            "image_path": str(image_path),
            "raw_hit": hit
        })
        
    # 2. Slide window and score
    windows: List[Dict[str, Any]] = []
    half_window = window_seconds / 2.0
    
    for video_id, items in video_groups.items():
        # Sort items by timestamp to ensure chronological order
        items.sort(key=lambda x: x["timestamp"])
        
        for anchor in items:
            anchor_time = anchor["timestamp"]
            window_start = max(0.0, anchor_time - half_window)
            window_end = anchor_time + half_window
            
            # Gather all hits in this video within the window boundaries
            in_window = [
                x for x in items 
                if window_start <= x["timestamp"] <= window_end
            ]
            
            if len(in_window) < min_hits_in_window:
                continue
                
            scores = [x["score"] for x in in_window]
            avg_score = float(np.mean(scores))
            peak_item = max(in_window, key=lambda x: x["score"])
            
            windows.append({
                "video_id": video_id,
                "window_score": avg_score,
                "peak_score": peak_item["score"],
                "window_start": window_start,
                "window_end": window_end,
                "representative_frame": peak_item["image_path"],
                "hit_count": len(in_window),
                "hits": [x["raw_hit"] for x in in_window]
            })
            
    # 3. Non-Maximum Suppression (NMS) to deduplicate overlapping windows
    # Sort windows by score descending so we select the highest scoring ones first
    windows.sort(key=lambda x: x["window_score"], reverse=True)
    
    deduped_windows: List[Dict[str, Any]] = []
    for w in windows:
        overlap = False
        for dw in deduped_windows:
            if dw["video_id"] == w["video_id"]:
                # Check for interval intersection
                max_start = max(w["window_start"], dw["window_start"])
                min_end = min(w["window_end"], dw["window_end"])
                if max_start < min_end:  # Non-empty intersection
                    overlap = True
                    break
        if not overlap:
            deduped_windows.append(w)
            
    return deduped_windows
