"""Shared fixtures. No real weights and no internet: the model is the same
architecture as production, built with random weights."""

from __future__ import annotations

import io
import json
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.model import build_model, save_checkpoint

ARCH = "densenet121"
TEST_THRESHOLD = 0.5


@pytest.fixture(scope="session")
def random_model():
    m = build_model(ARCH, pretrained=False)  # no download
    m.eval()
    return m


@pytest.fixture(scope="session")
def model_files(tmp_path_factory, random_model):
    d = tmp_path_factory.mktemp("models")
    model_path, card_path = d / "model.pt", d / "model_card.json"
    save_checkpoint(
        random_model,
        ARCH,
        model_path,
        threshold=TEST_THRESHOLD,
        class_names=["NORMAL", "PNEUMONIA"],
        img_size=224,
    )
    card = {
        "architecture": ARCH,
        "threshold": TEST_THRESHOLD,
        "test_metrics": {"auc": 0.0, "sensitivity": 0.0, "specificity": 0.0},
        "disclaimer": "test fixture",
    }
    card_path.write_text(json.dumps(card))
    return model_path, card_path, card


def _set_env(**kv):
    old = {k: os.environ.get(k) for k in kv}
    for k, v in kv.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)
    return old


@pytest.fixture(scope="session")
def client(model_files):
    model_path, card_path, _ = model_files
    old = _set_env(MODEL_PATH=model_path, MODEL_CARD_PATH=card_path, MODEL_URL=None)
    from api.main import app

    with TestClient(app) as c:  # runs lifespan -> loads the random-weight checkpoint
        yield c
    _set_env(**old)


def make_png(w: int = 300, h: int = 260, mode: str = "L") -> bytes:
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:h, 0:w]
    arr = ((xx + yy) % 256 * 0.7 + rng.integers(0, 60, (h, w))).astype(np.uint8)
    img = Image.fromarray(arr, mode="L").convert(mode)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
