# ============================================================
# AI Music Generator — Dockerfile
# ============================================================
# Multi-stage build:
#   1. builder — install Python deps
#   2. runtime — minimal image

FROM python:3.11-slim AS builder

WORKDIR /app

# System deps for audiocraft, demucs, whisper, soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ---- Runtime ----
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Create outputs directory
RUN mkdir -p backend/outputs

WORKDIR /app/backend

EXPOSE 8000

ENV MUSIC_GEN_HOST=0.0.0.0
ENV MUSIC_GEN_PORT=8000

CMD ["python", "main.py"]
