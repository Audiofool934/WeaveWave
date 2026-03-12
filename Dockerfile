# Multi-stage Dockerfile for WeaveWave
# Usage: docker build -t weavewave .

# --- Stage 1: Base image with system dependencies ---
FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Stage 2: Install Python dependencies ---
FROM base AS deps

COPY pyproject.toml requirements.txt ./
COPY repos/ repos/

RUN pip install --no-cache-dir -e ".[demo]" \
    && pip install --no-cache-dir -e repos/audiocraft || true

# --- Stage 3: Application ---
FROM deps AS app

COPY . .

RUN pip install --no-cache-dir -e .

EXPOSE 7860 8001

CMD ["weavewave-demo", "--listen", "0.0.0.0"]
