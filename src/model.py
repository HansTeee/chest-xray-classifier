"""Transfer-learning model factory.

The same `build_model(arch, pretrained=False)` is used by training, the API,
and the test fixture (random weights), so the architecture is defined once.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models

NUM_CLASSES = 2
SUPPORTED_ARCHS = ("densenet121", "efficientnet_b0", "resnet18")


def build_model(arch: str = "densenet121", pretrained: bool = True) -> nn.Module:
    if arch == "densenet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier = nn.Linear(m.classifier.in_features, NUM_CLASSES)
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
    elif arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    else:
        raise ValueError(f"Unsupported arch {arch!r}; choose from {SUPPORTED_ARCHS}")
    return m


def head_parameters(m: nn.Module, arch: str):
    return (m.fc if arch == "resnet18" else m.classifier).parameters()


def _blocks(m: nn.Module, arch: str) -> list[nn.Module]:
    """Backbone stages from shallow to deep (for partial unfreezing)."""
    if arch == "densenet121":
        f = m.features
        return [
            nn.ModuleList([f.conv0, f.norm0]),
            nn.ModuleList([f.denseblock1, f.transition1]),
            nn.ModuleList([f.denseblock2, f.transition2]),
            nn.ModuleList([f.denseblock3, f.transition3]),
            nn.ModuleList([f.denseblock4, f.norm5]),
        ]
    if arch == "efficientnet_b0":
        return list(m.features)
    if arch == "resnet18":
        return [nn.ModuleList([m.conv1, m.bn1]), m.layer1, m.layer2, m.layer3, m.layer4]
    raise ValueError(arch)


def set_trainable(m: nn.Module, arch: str, unfreeze_blocks: int | None) -> None:
    """unfreeze_blocks=None -> full fine-tune; k -> head + last k backbone blocks."""
    if unfreeze_blocks is None:
        for p in m.parameters():
            p.requires_grad = True
        return
    for p in m.parameters():
        p.requires_grad = False
    blocks = _blocks(m, arch)
    for b in blocks[len(blocks) - unfreeze_blocks :] if unfreeze_blocks > 0 else []:
        for p in b.parameters():
            p.requires_grad = True
    for p in head_parameters(m, arch):
        p.requires_grad = True


def gradcam_target_layer(m: nn.Module, arch: str) -> nn.Module:
    """Last convolutional feature layer (used by src/gradcam.py)."""
    if arch == "densenet121":
        return m.features.norm5  # output of denseblock4 + final BN, 7x7 at 224px
    if arch == "efficientnet_b0":
        return m.features[-1]
    if arch == "resnet18":
        return m.layer4
    raise ValueError(arch)


def save_checkpoint(m: nn.Module, arch: str, path, **meta) -> None:
    torch.save({"arch": arch, "state_dict": m.state_dict(), **meta}, path)


def load_checkpoint(path, map_location="cpu") -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    m = build_model(ckpt["arch"], pretrained=False)
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    return m, meta
