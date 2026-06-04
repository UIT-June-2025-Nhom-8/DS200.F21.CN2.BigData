"""
FaceGuard — FastAPI backend
Endpoints:
  GET  /api/health   — health check + device info
  POST /api/predict  — single-image inference (multipart/form-data)
"""

import io
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from inference import (
    CHECKPOINT_PATH,
    GradCAM,
    get_device,
    load_model,
    predict_image,
)

# ── App state ──────────────────────────────────────────────────────────────

_state: dict = {}

ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB

# ── Lifespan (load model once at startup) ─────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    device = get_device()
    print(f"[FaceGuard] Device     : {device}")

    if not CHECKPOINT_PATH.exists():
        raise RuntimeError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    model, transform = load_model(CHECKPOINT_PATH, device)
    gradcam = GradCAM(model)

    _state["device"]     = device
    _state["model"]      = model
    _state["gradcam"]    = gradcam
    _state["transform"]  = transform

    print(f"[FaceGuard] Model loaded: {CHECKPOINT_PATH.name}")
    print(f"[FaceGuard] Ready at    : http://localhost:8001")
    yield
    _state.clear()

# ── FastAPI app ────────────────────────────────────────────────────────────

app = FastAPI(
    title="FaceGuard API",
    description="AI-Generated Face Detection — EfficientNet-B0",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://localhost:4173",   # Vite preview
        "http://localhost:3000",
        "http://localhost:8001",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    device = _state.get("device", "unknown")
    return {
        "status":     "ok",
        "checkpoint": CHECKPOINT_PATH.name,
        "device":     str(device),
        "torch":      torch.__version__,
    }


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    gradcam: bool = True,
):
    # Validate MIME type
    if file.content_type not in ALLOWED_MIME:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Accepted: JPEG, PNG, WEBP.",
        )

    # Read bytes
    data = await file.read()
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 10 MB limit.")

    # Decode image
    try:
        pil_image = Image.open(io.BytesIO(data))
        pil_image.verify()
        pil_image = Image.open(io.BytesIO(data))  # re-open after verify()
    except (UnidentifiedImageError, Exception) as e:
        raise HTTPException(status_code=422, detail=f"Cannot decode image: {e}")

    # Run inference
    try:
        result = predict_image(
            pil_image=pil_image,
            model=_state["model"],
            gradcam=_state["gradcam"],
            device=_state["device"],
            transform=_state["transform"],
            include_gradcam=gradcam,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return JSONResponse(content=result)


# ── Serve built React frontend (production) ────────────────────────────────

_frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="static")
