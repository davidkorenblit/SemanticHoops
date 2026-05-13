# =============================================================================
# Semantic Sport-Tech — Dockerfile (Fixed Syntax)
# =============================================================================

# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System dependencies: Added ca-certificates for SSL and git for CLIP
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
    ca-certificates && \
    update-ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install heavy libraries (using cache layers)
RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121 \
    --trusted-host download.pytorch.org \
    --trusted-host files.pythonhosted.org \
    --trusted-host pypi.org \
    --trusted-host download-r2.pytorch.org
RUN pip install git+https://github.com/openai/CLIP.git \
    --trusted-host github.com \
    --trusted-host objects.githubusercontent.com

# Install remaining Python requirements
COPY requirements.txt .
RUN pip install -r requirements.txt


# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip python3-dev build-essential \
    ffmpeg libgl1 libglib2.0-0 git ca-certificates && \
    update-ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

COPY src/ ./src/
COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]