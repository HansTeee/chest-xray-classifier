"""Grad-CAM on the model's last convolutional layer.

Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
Gradient-based Localization" (ICCV 2017).

The heatmap shows which image regions most increased the score of the target
class. It is a coarse explanation (7x7 at 224px for DenseNet121), not a
segmentation, and it does not prove the model is looking at pathology.

Dependencies: torch, numpy, Pillow only (safe for the lean serving image).
"""

from __future__ import annotations

import base64
import io
import threading

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from src.model import gradcam_target_layer

_lock = threading.Lock()  # hooks are attached to a shared model; serialize calls


def predict_with_gradcam(
    model: nn.Module, arch: str, x: torch.Tensor, target_class: int | None = None
) -> tuple[torch.Tensor, np.ndarray, int]:
    """One forward + backward pass.

    Args:
        x: normalized 1x3xHxW tensor.
        target_class: class to explain; None = the predicted class (argmax).

    Returns:
        (softmax probabilities [C], cam HxW float32 in [0, 1] matching x's
        spatial size, explained class index)
    """
    if x.ndim != 4 or x.shape[0] != 1:
        raise ValueError(f"expected 1x3xHxW input, got {tuple(x.shape)}")
    layer = gradcam_target_layer(model, arch)
    store: dict[str, torch.Tensor] = {}

    def fwd_hook(_module, _inp, out):
        store["act"] = out
        out.register_hook(lambda g: store.__setitem__("grad", g))
        # Return a clone so downstream in-place ops (e.g. DenseNet's
        # F.relu(inplace=True)) cannot modify the saved activation.
        return out.clone()

    with _lock:
        model.eval()
        handle = layer.register_forward_hook(fwd_hook)
        try:
            with torch.enable_grad():
                x = x.detach().requires_grad_(False)
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[0].detach()
                cls = int(probs.argmax()) if target_class is None else int(target_class)
                model.zero_grad(set_to_none=True)
                logits[0, cls].backward()
        finally:
            handle.remove()
        act, grad = store["act"].detach(), store["grad"].detach()
        model.zero_grad(set_to_none=True)

    weights = grad.mean(dim=(2, 3), keepdim=True)  # global-average-pooled gradients
    cam = F.relu((weights * act).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = cam / cam.max() if cam.max() > 0 else cam
    return probs.cpu(), cam.cpu().numpy().astype(np.float32), cls


def _jet(v: np.ndarray) -> np.ndarray:
    """Jet colormap for values in [0, 1] -> uint8 HxWx3 (no matplotlib dependency)."""
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def overlay(image: Image.Image, cam: np.ndarray, alpha: float = 0.45, max_side: int = 768) -> Image.Image:
    """Blend a CAM (any size) onto the original X-ray (resized to <= max_side)."""
    img = image.convert("L")
    img.thumbnail((max_side, max_side))
    w, h = img.size
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    cam_r = np.asarray(cam_img, dtype=np.float32) / 255.0
    base = np.repeat(np.asarray(img, dtype=np.float32)[..., None], 3, axis=-1)
    heat = _jet(cam_r).astype(np.float32)
    a = alpha * cam_r[..., None]  # weak activations stay transparent
    out = (1 - a) * base + a * heat
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def to_base64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")
