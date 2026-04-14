# Semantic Sport-Tech

> **Text-to-video semantic search over sports footage** — powered by CLIP embeddings and Qdrant vector DB.

A production-ready ML pipeline that ingests hours of sports video, extracts semantically rich frames via adaptive optical-flow sampling, embeds them with OpenAI CLIP, and exposes a FastAPI search endpoint so you can query *"basketball slam dunk"* and retrieve the best matching clips with millisecond latency.

---

## Architecture

```
wcs_pro/
├── src/
│   ├── ingestion/          # Phase 2 — Video download & Smart Sampling
│   │   ├── downloader.py   #   yt-dlp wrapper
│   │   └── sampler.py      #   Adaptive FPS via Farneback optical flow
│   ├── models/             # Phase 3 — Plug-and-play embedding backends
│   │   ├── base.py         #   Abstract BaseEmbedder (swap CLIP → VideoMAE easily)
│   │   └── clip_wrapper.py #   CLIP ViT-B/32 … ViT-L/14 implementation
│   ├── db/                 # Phase 4 — Vector DB layer
│   │   ├── qdrant_client.py#   Connection management & collection bootstrap
│   │   └── indexer.py      #   Batched upsert & cosine similarity search
│   └── api/                # Phase 5 — FastAPI endpoints
│       ├── main.py         #   App factory + lifespan (startup/shutdown)
│       ├── schemas.py      #   Pydantic v2 request/response models
│       └── routers/
│           ├── health.py   #   GET  /health
│           ├── search.py   #   POST /search
│           └── ingest.py   #   POST /ingest
├── data/
│   ├── raw/                # Downloaded videos (.gitignore'd)
│   └── processed/          # Extracted frames (.gitignore'd)
├── tests/
│   └── test_health.py
├── Dockerfile              # Multi-stage CUDA build
├── docker-compose.yml      # App + Qdrant services
├── requirements.txt
└── .env.example
```

## Quickstart

### 1. Prerequisites
- Docker + Docker Compose ≥ v2
- NVIDIA GPU + nvidia-container-toolkit (optional — comment out GPU section in `docker-compose.yml` for CPU)

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env — set CLIP_DEVICE=cpu if no GPU
```

### 3. Launch Services
```bash
docker compose up --build
```

- **API**: http://localhost:8000/docs (Swagger UI)
- **Qdrant Dashboard**: http://localhost:6333/dashboard

### 4. Ingest a Video
```bash
curl -X POST http://localhost:8000/ingest/ \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=<VIDEO_ID>", "sport": "basketball"}'
```

### 5. Search
```bash
curl -X POST http://localhost:8000/search/ \
  -H "Content-Type: application/json" \
  -d '{"query": "basketball slam dunk", "top_k": 5}'
```

---

## Development (without Docker)

```bash
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
uvicorn main:app --reload
```

Run tests:
```bash
pytest tests/ -v
```

---

## Implementation Roadmap

| Phase | Weeks | Status |
|-------|-------|--------|
| 1 — Infrastructure (Docker + Qdrant) | 1 | ✅ Done |
| 2 — Smart Sampling Pipeline | 2 | 🔧 In Progress |
| 3 — CLIP Embedding Layer | 3 | 🔲 Planned |
| 4 — Retrieval Engine | 4 | 🔲 Planned |
| 5 — API + Evaluation | 5 | 🔲 Planned |
| 6 — Streamlit UI + Docs | 6 | 🔲 Planned |

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Smart Sampling (optical flow)** | Reduces 162k frames/90min match → ~5–15% while preserving semantic information |
| **Plug-and-play BaseEmbedder** | Swap CLIP → VideoMAE → InternVideo without touching API or DB code |
| **Qdrant over Pinecone** | Self-hosted, HNSW index, filtering support, no per-vector pricing |
| **Multi-stage Dockerfile** | Builder stage installs all deps; runtime stage is lean (no build tools) |
| **Named Docker volumes** | Qdrant data survives `docker compose down`; never enters Git history |

---

## Git Safety

This repo **never** stores:
- `.mp4`, `.avi`, or any video files
- Model weights (`.pt`, `.pth`, `.bin`, `.safetensors`)
- Qdrant storage volumes (`qdrant_storage/`)
- Secrets (`.env`)

All enforced via [`.gitignore`](.gitignore).
