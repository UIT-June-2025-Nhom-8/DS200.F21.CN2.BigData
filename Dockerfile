# ── Stage 1: Build React frontend ────────────────────────────────────────────
FROM node:20-slim AS frontend-builder

WORKDIR /build
COPY app/frontend/package*.json ./
RUN npm ci
COPY app/frontend/ ./
RUN npm run build

# ── Stage 2: Python runtime ───────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

COPY app/backend/requirements.txt ./
RUN grep -vE "^torch" requirements.txt | \
    pip install --no-cache-dir -r /dev/stdin

COPY src/ ./src/
COPY artifacts/checkpoints/best_stage3.pth ./artifacts/checkpoints/best_stage3.pth
COPY app/backend/ ./app/backend/

COPY --from=frontend-builder /build/dist/ ./app/frontend/dist/

EXPOSE 8001

WORKDIR /workspace/app/backend
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8001}
