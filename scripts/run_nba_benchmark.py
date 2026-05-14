"""scripts/run_nba_benchmark.py
=================================================
QA Benchmarking Suite – Zero-Shot CLIP Evaluation

Iterates through a taxonomy of basketball queries (Tier 1 → Tier 3),
retrieves the Top-5 nearest-neighbour frames from the `nba_frames`
Qdrant collection, copies them into a structured output directory, and
generates a manual-annotation template (notes.txt) in each folder.

Usage (inside the Docker container or local venv):
    python scripts/run_nba_benchmark.py

Environment variables (matching docker-compose.yml defaults):
    QDRANT_HOST  – defaults to "qdrant"
    QDRANT_PORT  – defaults to 6333

Output layout:
    data/QA_results/
    └── tier1_static/
    │   └── player_dunking/
    │       ├── rank_1_score_0.2900.jpg
    │       ├── ...
    │       └── notes.txt
    └── tier2_dynamic/
    └── tier3_tactical/
"""

import os
import re
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime, timezone

from qdrant_client import QdrantClient
from src.models.embeddings import CLIPWrapper

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

COLLECTION_NAME = "nba_frames"
TOP_K = 5

# Root of the repo (one level above /scripts)
REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "data" / "QA_results"

# ---------------------------------------------------------------------------
# Configuration Loading
# ---------------------------------------------------------------------------

def load_queries(config_path: Path) -> list:
    """Load queries from the JSON configuration file."""
    if not config_path.exists():
        print(f"[ERROR] Configuration file not found at: {config_path}")
        print("Please ensure the configs/nba_queries.json file exists.")
        sys.exit(1)
        
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        print(f"[ERROR] Failed to parse JSON config: {exc}")
        sys.exit(1)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slugify(text: str) -> str:
    """Convert a display name or query string to a filesystem-safe slug."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text)
    return text.strip("_")


def write_notes_template(folder: Path, query_string: str, hits: list) -> None:
    """Write a manual-annotation template into the query result folder."""
    lines = [
        "=" * 60,
        "QA BENCHMARK – Manual Annotation Sheet",
        "=" * 60,
        f"Query        : {query_string}",
        f"Generated at : {datetime.now(timezone.utc).isoformat()}",
        f"Top-K        : {TOP_K}",
        "",
        "Instructions:",
        "  For each rank, assign one label:",
        "    CORRECT      – Frame clearly matches the query",
        "    PHASE_ERROR  – Right action, but wrong temporal phase",
        "    WRONG        – Completely unrelated content",
        "",
        "-" * 60,
        f"{'Rank':<6} {'Score':<10} {'Label':<14} {'Notes'}",
        "-" * 60,
    ]

    for rank, hit in enumerate(hits, start=1):
        score = f"{hit.score:.4f}"
        lines.append(f"{rank:<6} {score:<10} {'[ ]':<14} ")

    lines += [
        "-" * 60,
        "",
        f"Precision@5 (CORRECT / {TOP_K}): ___",
        "",
        "Overall notes / observations:",
        "  ",
        "",
    ]

    notes_path = folder / "notes.txt"
    notes_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark() -> None:
    # ── Load Configuration ─────────────────────────────────────────────────
    config_path = REPO_ROOT / "configs" / "nba_queries.json"
    queries = load_queries(config_path)

    # ── Connect to Qdrant ──────────────────────────────────────────────────
    host = os.getenv("QDRANT_HOST", "qdrant")
    port = int(os.getenv("QDRANT_PORT", 6333))
    print(f"[benchmark] Connecting to Qdrant at {host}:{port} …")
    client = QdrantClient(host=host, port=port)

    # ── Load CLIP model (once, shared across all queries) ──────────────────
    print("[benchmark] Loading CLIP model …")
    model = CLIPWrapper()
    print("[benchmark] Model ready.\n")

    # ── Tracking counters ──────────────────────────────────────────────────
    total_queries = len(queries)
    processed = 0
    skipped_frames = 0  # frames whose source path was missing on disk

    # ── Iterate through queries ────────────────────────────────────────────
    for q_obj in queries:
        tier_slug = q_obj["tier"]
        display_name = q_obj["name"]
        query_string = q_obj["query"]
        print(f"[{tier_slug}] Query: \"{query_string}\"")

        # 1. Generate text embedding
        query_vector = model.get_text_embedding(query_string)
        if hasattr(query_vector, "tolist"):
            query_vector = query_vector.tolist()

        # 2. Search Qdrant
        try:
            hits = client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=TOP_K,
                with_payload=True,
            )
        except Exception as exc:
            print(f"  [ERROR] Qdrant search failed: {exc}  – skipping query.\n")
            continue

        if not hits:
            print("  [WARN] No results returned – skipping query.\n")
            continue

        # 3. Prepare output directory
        query_dir = OUTPUT_ROOT / tier_slug / display_name
        query_dir.mkdir(parents=True, exist_ok=True)

        # 4. Copy and rename result frames
        for rank, hit in enumerate(hits, start=1):
            score = hit.score
            payload = hit.payload or {}
            raw_path = payload.get("image_path")

            if not raw_path:
                print(f"  [WARN] Rank {rank}: payload has no 'image_path' key.")
                skipped_frames += 1
                continue

            # image_path in the payload may be relative to the repo root
            source_path = Path(raw_path)
            if not source_path.is_absolute():
                source_path = REPO_ROOT / source_path

            dest_filename = f"rank_{rank}_score_{score:.4f}.jpg"
            dest_path = query_dir / dest_filename

            if not source_path.exists():
                print(f"  [WARN] Rank {rank}: source frame not found on disk: {source_path}")
                # Write a placeholder so the folder still reflects the result
                dest_path.write_text(
                    f"MISSING SOURCE\nOriginal path: {raw_path}\nScore: {score:.4f}\n",
                    encoding="utf-8",
                )
                skipped_frames += 1
                continue

            try:
                shutil.copy2(source_path, dest_path)
                print(f"  Rank {rank} | score={score:.4f} | {source_path.name} → {dest_filename}")
            except OSError as exc:
                print(f"  [ERROR] Rank {rank}: failed to copy frame: {exc}")
                skipped_frames += 1
                continue

        # 5. Write annotation template
        write_notes_template(query_dir, query_string, hits)
        print(f"  → Results saved to: {query_dir.relative_to(REPO_ROOT)}\n")
        processed += 1

    # ── Final summary ──────────────────────────────────────────────────────
    print("=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print(f"  Queries processed   : {processed} / {total_queries}")
    print(f"  Frames skipped      : {skipped_frames}  (missing on disk or no path in payload)")
    print(f"  Results directory   : {OUTPUT_ROOT.relative_to(REPO_ROOT)}")
    print("")
    print("Next step: Open data/QA_results/ and fill in the notes.txt")
    print("files for each query to compute Precision@5 per tier.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_benchmark()
