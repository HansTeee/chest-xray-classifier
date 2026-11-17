from __future__ import annotations

import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import api.main as api_main
from tests.conftest import TEST_THRESHOLD, _set_env, make_png


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert "not a medical device" in body["disclaimer"].lower()


def test_model_info(client, model_files):
    r = client.get("/model-info")
    assert r.status_code == 200
    assert r.json() == model_files[2]


def test_predict_synthetic_image(client):
    r = client.post("/predict", files={"file": ("xray.png", make_png(300, 260), "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["label"] in {"NORMAL", "PNEUMONIA"}
    assert 0.0 <= body["pneumonia_probability"] <= 1.0
    assert body["threshold"] == TEST_THRESHOLD
    assert body["label"] == ("PNEUMONIA" if body["pneumonia_probability"] >= TEST_THRESHOLD else "NORMAL")
    assert body["gradcam_class"] == body["label"]
    assert body["inference_ms"] > 0
    assert body["model_arch"] == "densenet121"
    overlay = Image.open(io.BytesIO(base64.b64decode(body["gradcam_png_base64"])))
    assert overlay.format == "PNG"
    assert overlay.size == (300, 260)  # overlay matches the uploaded image
    assert overlay.mode == "RGB"


def test_predict_accepts_rgb_jpeg(client):
    buf = io.BytesIO()
    Image.open(io.BytesIO(make_png(256, 256, mode="RGB"))).save(buf, format="JPEG")
    r = client.post("/predict", files={"file": ("x.jpg", buf.getvalue(), "image/jpeg")})
    assert r.status_code == 200, r.text


def test_predict_rejects_non_image(client):
    r = client.post("/predict", files={"file": ("notes.png", b"definitely not an image", "image/png")})
    assert r.status_code == 415
    assert "not a readable image" in r.json()["detail"]


def test_predict_rejects_empty_file(client):
    r = client.post("/predict", files={"file": ("empty.png", b"", "image/png")})
    assert r.status_code == 400


def test_predict_rejects_oversized_file(client, monkeypatch):
    monkeypatch.setattr(api_main, "MAX_UPLOAD_BYTES", 1000)
    r = client.post("/predict", files={"file": ("big.png", make_png(300, 300), "image/png")})
    assert r.status_code == 413


def test_predict_rejects_tiny_image(client):
    r = client.post("/predict", files={"file": ("tiny.png", make_png(16, 16), "image/png")})
    assert r.status_code == 422


def test_predict_requires_file(client):
    assert client.post("/predict").status_code == 422


def test_openapi_has_disclaimer(client):
    desc = client.get("/openapi.json").json()["info"]["description"]
    assert "NOT a medical device" in desc


def test_missing_model_without_repo_degrades_gracefully(tmp_path):
    old = _set_env(
        MODEL_PATH=tmp_path / "missing.pt", MODEL_CARD_PATH=tmp_path / "missing.json", HF_MODEL_REPO=None
    )
    try:
        with TestClient(api_main.app) as c:
            h = c.get("/health").json()
            assert h["model_loaded"] is False and h["status"] == "degraded"
            assert "HF_MODEL_REPO" in h["error"]
            assert c.post("/predict", files={"file": ("x.png", make_png(), "image/png")}).status_code == 503
            assert c.get("/model-info").status_code == 503
    finally:
        _set_env(**old)
        api_main.load_state(api_main.app)  # restore the shared app for other tests


@pytest.mark.parametrize("path", ["/docs", "/openapi.json"])
def test_docs_available(client, path):
    assert client.get(path).status_code == 200
