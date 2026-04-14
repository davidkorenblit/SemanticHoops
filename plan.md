# Project Implementation Plan - Semantic Sport-Tech

## Phase 1: Data & Infrastructure (Week 1)
* **Tasks:**
    * Setup Docker environment (Python + GPU support + Vector DB).
    * Source selection: DeepSportRadar vs. YouTube (`yt-dlp`).
    * Initial Engineering Justification doc.
* **Deliverable:** Working containerized environment.

## Phase 2: Smart Sampling & Pipeline (Week 2)
* **Tasks:**
    * Build video-to-frame ingestion engine.
    * Implement Smart Sampling logic (Adaptive FPS).
    * Measure processing latency per video hour.
* **Deliverable:** Efficient data pipeline.

## Phase 3: Embedding Layer (Week 3)
* **Tasks:**
    * Integrate CLIP / Multimodal embeddings.
    * Create modular "Model Wrapper" for easy swapping.
    * Validation of embedding quality on sample clips.
* **Deliverable:** Feature extraction module.

## Phase 4: Retrieval Engine & Vector DB (Week 4)
* **Tasks:**
    * Deploy and configure Vector DB (e.g., Qdrant).
    * Index initial dataset.
    * Implement Cosine Similarity search.
* **Deliverable:** Functional search backend.

## Phase 5: API & Performance Optimization (Week 5)
* **Tasks:**
    * Develop FastAPI backend for the search engine.
    * Run evaluation metrics (mAP, Recall@K).
    * Optimization pass on inference speed.
* **Deliverable:** Production-grade API + Evaluation Report.

## Phase 6: UI & Final Polish (Week 6)
* **Tasks:**
    * Build Streamlit frontend for visual demo.
    * Complete README.md and engineering documentation.
    * Prepare for "Hard Questions" mock interview.
* **Deliverable:** Portfolio-ready MVP.
