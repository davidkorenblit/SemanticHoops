# =============================================================================
# Semantic Sport-Tech — Dockerfile
# Multi-stage build: keeps the final image lean by separating build deps.
# Base: CUDA 12.1 + cuDNN8 for GPU-accelerated CLIP inference.
# =============================================================================

# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System dependencies (ffmpeg for OpenCV video decode, git for pip VCS deps)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install Python deps into an isolated venv so we can copy it cleanly
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt


# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the pre-built venv from builder
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Copy application source (data/ volumes are mounted at runtime, not baked in)
COPY src/ ./src/
COPY main.py .

# Non-root user for security hardening
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
USER appuser

EXPOSE 8000

# Entrypoint: uvicorn serves the FastAPI app
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
