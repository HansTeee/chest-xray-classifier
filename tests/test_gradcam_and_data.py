from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from src.data import gpu_augment, patient_id, preprocess_for_inference
from src.gradcam import overlay, predict_with_gradcam
from tests.conftest import ARCH


@pytest.mark.parametrize("hw", [(224, 224), (256, 192)])
def test_gradcam_matches_input_dimensions(random_model, hw):
    torch.manual_seed(0)
    x = torch.randn(1, 3, *hw)
    probs, cam, cls = predict_with_gradcam(random_model, ARCH, x)
    assert cam.shape == hw
    assert cam.dtype == np.float32
    assert cam.min() >= 0.0 and cam.max() <= 1.0
    assert probs.shape == (2,) and torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
    assert cls == int(probs.argmax())


def test_gradcam_target_class_and_no_leftover_hooks(random_model):
    x = torch.randn(1, 3, 224, 224)
    _, _, cls = predict_with_gradcam(random_model, ARCH, x, target_class=1)
    assert cls == 1
    layer = random_model.features.norm5
    assert not layer._forward_hooks  # hook removed after each call
    assert all(p.grad is None for p in random_model.parameters())


def test_gradcam_rejects_batched_input(random_model):
    with pytest.raises(ValueError):
        predict_with_gradcam(random_model, ARCH, torch.randn(2, 3, 224, 224))


def test_overlay_matches_image_size():
    img = Image.new("L", (400, 300), 128)
    out = overlay(img, np.random.rand(224, 224).astype(np.float32))
    assert out.size == (400, 300) and out.mode == "RGB"


def test_preprocess_shape(tmp_path):
    p = tmp_path / "x.png"
    Image.new("L", (500, 400), 100).save(p)
    x = preprocess_for_inference(p)
    assert x.shape == (1, 3, 224, 224) and x.dtype == torch.float32


@pytest.mark.parametrize(
    "name,cls,expected",
    [
        ("person123_bacteria_456.jpeg", "PNEUMONIA", "PNEUMONIA:person123"),
        ("person123_virus_9.jpeg", "PNEUMONIA", "PNEUMONIA:person123"),
        ("IM-0115-0001.jpeg", "NORMAL", "NORMAL:IM-0115"),
        ("NORMAL2-IM-0383-0001.jpeg", "NORMAL", "NORMAL:NORMAL2-IM-0383"),
    ],
)
def test_patient_id_groups_images_by_patient(name, cls, expected):
    assert patient_id(name, cls) == expected


def test_gpu_augment_is_not_a_horizontal_flip():
    # Left half bright, right half dark: a flip would invert the mean difference.
    x = torch.zeros(16, 3, 224, 224, dtype=torch.uint8)
    x[..., :112] = 200
    out = gpu_augment(x)
    assert out.shape == (16, 3, 224, 224)
    left, right = out[..., :100].mean(dim=(1, 2, 3)), out[..., 124:].mean(dim=(1, 2, 3))
    assert bool((left > right).all())
