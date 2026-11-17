"""FastAPI inference service for the chest X-ray pneumonia classifier.

Run locally (CMD):
    .venv\\Scripts\\activate
    uvicorn api.main:app --port 8000
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import time
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from src.data import CLASSES, preprocess_for_inference
from src.gradcam import overlay, predict_with_gradcam, to_base64_png
from src.model import load_checkpoint

log = logging.getLogger("api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

DISCLAIMER = (
    "EDUCATIONAL DEMO ONLY. This is NOT a medical device and must NOT be used for "
    "clinical diagnosis, screening, or any clinical decision."
)

API_DESCRIPTION = f"""
> ⚠️ **{DISCLAIMER}**

Binary classifier for pediatric chest X-rays (**NORMAL** vs **PNEUMONIA**),
DenseNet121 transfer learning. Every prediction includes a **Grad-CAM**
heatmap showing which regions most influenced the decision.

The heatmap is a coarse explanation, not a lesion segmentation. The model was
trained on a single pediatric dataset (Kermany et al., CC BY 4.0, ages 1-5,
one hospital) and has **not been clinically validated**. On the held-out test
set it has high sensitivity but only ~0.74 specificity, so false alarms are common.
"""

MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "10"))
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)
MIN_SIDE_PX = 32
MAX_PIXELS = 50_000_000  # decompression-bomb guard


def _paths() -> tuple[Path, Path]:
    return (
        Path(os.getenv("MODEL_PATH", "models/model.pt")),
        Path(os.getenv("MODEL_CARD_PATH", "models/model_card.json")),
    )


def _fetch(url: str, dest: Path, sha256: str | None = None, timeout: int = 120) -> None:
    """Download url -> dest atomically, optionally verifying a SHA-256 digest."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    h = hashlib.sha256()
    req = urllib.request.Request(url, headers={"User-Agent": "chest-xray-classifier"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            h.update(chunk)
            f.write(chunk)
    if sha256 and h.hexdigest().lower() != sha256.lower():
        tmp.unlink(missing_ok=True)
        raise ValueError(f"SHA-256 mismatch for {url}: got {h.hexdigest()}, expected {sha256}")
    tmp.replace(dest)


def _download_weights(model_path: Path, card_path: Path) -> None:
    """Download weights (and the model card) from MODEL_URL / MODEL_CARD_URL,
    e.g. GitHub Release assets. Weights are never committed to git."""
    url = os.getenv("MODEL_URL")
    if not url:
        raise FileNotFoundError(f"{model_path} not found and MODEL_URL is not set")
    log.info("Downloading weights from %s", url)
    _fetch(url, model_path, os.getenv("MODEL_SHA256") or None)
    card_url = os.getenv("MODEL_CARD_URL")
    if card_url and not card_path.exists():
        try:
            _fetch(card_url, card_path)
        except Exception:
            log.warning("Could not download model card from %s", card_url, exc_info=True)


def load_state(app: FastAPI) -> None:
    app.state.model, app.state.meta, app.state.card = None, {}, {}
    app.state.load_error = None
    model_path, card_path = _paths()
    try:
        if not model_path.exists():
            _download_weights(model_path, card_path)
        torch.set_num_threads(max(1, int(os.getenv("TORCH_THREADS", os.cpu_count() or 1))))
        model, meta = load_checkpoint(model_path, map_location="cpu")
        app.state.model, app.state.meta = model, meta
        if card_path.exists():
            app.state.card = json.loads(card_path.read_text())
        log.info("Model loaded: arch=%s threshold=%s", meta.get("arch"), meta.get("threshold"))
    except Exception as e:  # keep serving /health so the failure is visible
        app.state.load_error = f"{type(e).__name__}: {e}"
        log.exception("Model failed to load")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_state(app)
    yield


app = FastAPI(
    title="Chest X-ray Pneumonia Classifier API",
    version="1.0.0",
    description=API_DESCRIPTION,
    lifespan=lifespan,
)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    error: str | None = None
    disclaimer: str = DISCLAIMER


class PredictResponse(BaseModel):
    label: str = Field(description="NORMAL or PNEUMONIA (probability >= threshold)")
    pneumonia_probability: float = Field(ge=0, le=1)
    threshold: float = Field(description="Decision threshold, chosen on the validation set")
    inference_ms: float = Field(description="Model forward + Grad-CAM time in milliseconds")
    gradcam_class: str = Field(description="Class the heatmap explains (the predicted label)")
    gradcam_png_base64: str = Field(description="Grad-CAM heatmap overlaid on the X-ray, base64 PNG")
    model_arch: str
    disclaimer: str = DISCLAIMER


def _threshold(request: Request) -> float:
    meta, card = request.app.state.meta, request.app.state.card
    return float(meta.get("threshold", card.get("threshold", 0.5)))


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health(request: Request) -> HealthResponse:
    loaded = request.app.state.model is not None
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        error=request.app.state.load_error,
    )


@app.get("/model-info", tags=["service"])
def model_info(request: Request) -> dict[str, Any]:
    card = request.app.state.card
    if not card:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model card not available")
    return card


def _read_image(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()  # cheap integrity check; invalidates img
        img = Image.open(io.BytesIO(data))
        if img.width * img.height > MAX_PIXELS:
            raise HTTPException(413, "Image dimensions too large")
        img.load()
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as e:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            "File is not a readable image. Upload a PNG or JPEG chest X-ray.",
        ) from e
    if min(img.size) < MIN_SIDE_PX:
        raise HTTPException(422, f"Image too small (min side {MIN_SIDE_PX}px)")
    return img


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["inference"],
    responses={
        400: {"description": "Empty upload"},
        413: {"description": "File larger than the upload limit"},
        415: {"description": "Not an image"},
        422: {"description": "Image too small / malformed request"},
        503: {"description": "Model not loaded"},
    },
)
async def predict(request: Request, file: UploadFile = File(..., description="Chest X-ray (PNG/JPEG)")):
    model = request.app.state.model
    if model is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model not loaded")

    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Empty file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File exceeds {MAX_UPLOAD_MB:g} MB limit")
    img = _read_image(data)

    arch = request.app.state.meta.get("arch", "densenet121")
    thr = _threshold(request)
    t0 = time.perf_counter()
    x = preprocess_for_inference(io.BytesIO(data))
    probs, cam, _ = predict_with_gradcam(model, arch, x, target_class=None)
    p = float(probs[1])
    label = CLASSES[1] if p >= thr else CLASSES[0]
    # Explain the decision actually made (thresholded label, not argmax)
    if CLASSES.index(label) != int(probs.argmax()):
        _, cam, _ = predict_with_gradcam(model, arch, x, target_class=CLASSES.index(label))
    inference_ms = (time.perf_counter() - t0) * 1000

    return PredictResponse(
        label=label,
        pneumonia_probability=round(p, 6),
        threshold=round(thr, 6),
        inference_ms=round(inference_ms, 1),
        gradcam_class=label,
        gradcam_png_base64=to_base64_png(overlay(img, cam)),
        model_arch=arch,
    )
