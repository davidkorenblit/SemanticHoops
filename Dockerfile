# =============================================================================
# Semantic Sport-Tech — Dockerfile
# Multi-stage build for a lean production-ready image.
# Using Python 3.10 (Ubuntu 22.04 default) for maximum stability.
# =============================================================================

# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System dependencies: Added python3-dev and build-essential for C-extensions in pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-pip \
    python3-dev \
    build-essential \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Create virtual environment using the system's python3
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python requirements
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt


# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Install only necessary runtime system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the pre-built virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Copy application source code
COPY src/ ./src/
COPY main.py .

# Permission Fix: Run as root for local development to ensure write access to volumes.
# For production, we would uncomment the lines below.
# RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
# USER appuser

EXPOSE 8000

# Entrypoint: uvicorn serves the FastAPI app with 1 worker for GPU stability.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]