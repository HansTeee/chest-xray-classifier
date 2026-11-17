"""Dataset normalization, patient-level train/val split, transforms, loaders.

CLI (CMD):
    python -m src.data prepare --src "<kagglehub path>"
    python -m src.data split
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

CLASSES = ("NORMAL", "PNEUMONIA")  # index 1 = positive class
IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_EXTS = {".jpeg", ".jpg", ".png"}

DATA_ROOT = Path("data/chest_xray")
CACHE_DIR = Path("data/cache")
MANIFEST = DATA_ROOT / "manifest.csv"

_PNEU_RE = re.compile(r"^(person\d+)_")
_NORMAL_RE = re.compile(r"^(.*)-\d+\.jpe?g$", re.IGNORECASE)


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------
def _find_split_root(src: Path) -> Path:
    """Return the shallowest folder containing train/ and test/ (skips the
    duplicated nested chest_xray/chest_xray copy and __MACOSX)."""
    candidates = [
        p.parent
        for p in src.rglob("train")
        if p.is_dir() and (p.parent / "test").is_dir() and "__MACOSX" not in p.parts
    ]
    if not candidates:
        raise FileNotFoundError(f"No folder with train/ and test/ under {src}")
    return min(candidates, key=lambda p: len(p.parts))


def prepare(src: Path, dst: Path = DATA_ROOT) -> None:
    """Copy the raw download into a clean data/chest_xray/{train,test}/{CLASS}.

    The official 16-image val/ split is folded into train; a proper
    patient-level validation set is carved out by `split()`.
    """
    root = _find_split_root(src)
    print(f"Using split root: {root}")
    if dst.exists():
        shutil.rmtree(dst)
    for split_in, split_out in (("train", "train"), ("val", "train"), ("test", "test")):
        for cls in CLASSES:
            in_dir = root / split_in / cls
            if not in_dir.is_dir():
                continue
            out_dir = dst / split_out / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            for f in in_dir.iterdir():
                if f.is_file() and f.suffix.lower() in IMG_EXTS and not f.name.startswith("."):
                    shutil.copy2(f, out_dir / f.name)
    # Remove any junk that slipped through
    for junk in list(dst.rglob("__MACOSX")) + list(dst.rglob(".DS_Store")):
        shutil.rmtree(junk) if junk.is_dir() else junk.unlink()
    print_counts(dst)


# --------------------------------------------------------------------------
# Patient-level split
# --------------------------------------------------------------------------
def patient_id(filename: str, cls: str) -> str:
    """Patient key from a filename.

    PNEUMONIA: person123_bacteria_456.jpeg -> person123 (bacteria and virus
    images of the same personN are grouped together, which is conservative).
    NORMAL: IM-0115-0001.jpeg -> IM-0115; NORMAL2-IM-0383-0001.jpeg -> NORMAL2-IM-0383.
    """
    if cls == "PNEUMONIA":
        m = _PNEU_RE.match(filename)
    else:
        m = _NORMAL_RE.match(filename)
    if not m:
        raise ValueError(f"Cannot parse patient id from {filename!r}")
    return f"{cls}:{m.group(1)}"


def split(root: Path = DATA_ROOT, val_frac: float = 0.15, seed: int = 42) -> None:
    """Move ~val_frac of train PATIENTS (stratified by class) into val/."""
    rng = random.Random(seed)
    val_dir = root / "val"
    if val_dir.exists():
        raise RuntimeError(f"{val_dir} already exists; re-run `prepare` first.")

    for cls in CLASSES:
        groups: dict[str, list[Path]] = defaultdict(list)
        for f in sorted((root / "train" / cls).iterdir()):
            groups[patient_id(f.name, cls)].append(f)
        patients = sorted(groups)
        rng.shuffle(patients)
        n_images = sum(len(v) for v in groups.values())
        target, taken = val_frac * n_images, 0
        (val_dir / cls).mkdir(parents=True, exist_ok=True)
        for pid in patients:
            if taken >= target:
                break
            for f in groups[pid]:
                shutil.move(str(f), val_dir / cls / f.name)
                taken += 1

    write_manifest(root)
    verify_no_patient_leakage(root)
    print_counts(root)


def write_manifest(root: Path = DATA_ROOT) -> None:
    rows = []
    for split_name in ("train", "val", "test"):
        for label, cls in enumerate(CLASSES):
            d = root / split_name / cls
            if not d.is_dir():
                continue
            for f in sorted(d.iterdir()):
                rows.append(
                    {
                        "path": f.as_posix(),
                        "split": split_name,
                        "label": label,
                        "class": cls,
                        "patient_id": patient_id(f.name, cls),
                    }
                )
    with open(root / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_manifest(path: Path = MANIFEST) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def verify_no_patient_leakage(root: Path = DATA_ROOT) -> None:
    rows = read_manifest(root / "manifest.csv")
    by_split: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        by_split[r["split"]].add(r["patient_id"])
    overlap = by_split["train"] & by_split["val"]
    if overlap:
        raise AssertionError(f"{len(overlap)} patients in both train and val: {sorted(overlap)[:5]}")
    print("OK: no patient appears in both train and val.")
    tt = by_split["train"] & by_split["test"]
    if tt:
        print(
            f"NOTE: {len(tt)} patient-ID strings occur in both train and the official test set "
            "(never with the same pathogen type; may be ID reuse, cannot be verified)."
        )


def print_counts(root: Path = DATA_ROOT) -> None:
    print(f"\n{'split':<6} {'NORMAL':>7} {'PNEU':>7} {'total':>7} {'%pneu':>6} {'patients':>9}")
    for split_name in ("train", "val", "test"):
        counts, pats = [], set()
        for cls in CLASSES:
            d = root / split_name / cls
            files = list(d.iterdir()) if d.is_dir() else []
            counts.append(len(files))
            pats |= {patient_id(f.name, cls) for f in files}
        if sum(counts) == 0:
            continue
        tot = sum(counts)
        print(
            f"{split_name:<6} {counts[0]:>7} {counts[1]:>7} {tot:>7} "
            f"{100 * counts[1] / tot:>5.1f}% {len(pats):>9}"
        )


# --------------------------------------------------------------------------
# Tensors, transforms, loaders
# --------------------------------------------------------------------------
def load_image(path_or_file, size: int = IMG_SIZE) -> torch.Tensor:
    """Load any image as a uint8 3xHxW tensor (grayscale replicated to RGB)."""
    img = Image.open(path_or_file).convert("L").resize((size, size), Image.BILINEAR)
    arr = np.array(img, dtype=np.uint8)
    return torch.from_numpy(arr).unsqueeze(0).expand(3, -1, -1).contiguous()


def normalize_transform() -> v2.Compose:
    return v2.Compose([v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def gpu_augment(x: torch.Tensor, max_rot_deg: float = 10.0, max_shift: float = 0.05,
                max_scale: float = 0.05, jitter: float = 0.2) -> torch.Tensor:
    """Batched light augmentation on-device, per-sample random params.

    uint8 Bx3xHxW -> normalized float. Small rotation/shift/scale and
    brightness/contrast. NO horizontal flips (heart position matters).
    """
    x = x.float() / 255.0
    b, dev = x.shape[0], x.device

    def u(lo, hi):
        return torch.empty(b, device=dev).uniform_(lo, hi)

    ang = torch.deg2rad(u(-max_rot_deg, max_rot_deg))
    s = u(1 - max_scale, 1 + max_scale)
    tx, ty = u(-2 * max_shift, 2 * max_shift), u(-2 * max_shift, 2 * max_shift)  # [-1,1] coords
    cos, sin = torch.cos(ang) / s, torch.sin(ang) / s
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = torch.nn.functional.affine_grid(theta, list(x.shape), align_corners=False)
    x = torch.nn.functional.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    x = x * u(1 - jitter, 1 + jitter).view(b, 1, 1, 1)  # brightness
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * u(1 - jitter, 1 + jitter).view(b, 1, 1, 1) + mean  # contrast
    x = x.clamp(0, 1)
    m = torch.tensor(IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
    sd = torch.tensor(IMAGENET_STD, device=dev).view(1, 3, 1, 1)
    return (x - m) / sd


def preprocess_for_inference(path_or_file) -> torch.Tensor:
    """Image -> normalized 1x3x224x224 float tensor (same as eval preprocessing)."""
    return normalize_transform()(load_image(path_or_file)).unsqueeze(0)


def _cached_split(split_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode + resize every image once and cache as uint8 to data/cache/."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{split_name}_{IMG_SIZE}.pt"
    rows = [r for r in read_manifest() if r["split"] == split_name]
    paths = [r["path"] for r in rows]
    if cache.exists():
        blob = torch.load(cache, weights_only=True)
        if blob.get("paths") == paths:
            return blob["x"], blob["y"]
    x = torch.stack([load_image(p) for p in paths])
    y = torch.tensor([int(r["label"]) for r in rows], dtype=torch.long)
    torch.save({"x": x, "y": y, "paths": paths}, cache)
    return x, y


class XrayTensorDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, transform):
        self.x, self.y, self.transform = x, y, transform

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i):
        return self.transform(self.x[i]), self.y[i]


def get_split_tensors(split_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    return _cached_split(split_name)


def make_loader(split_name: str, batch_size: int = 32, train: bool = False) -> DataLoader:
    """train=True yields raw uint8 batches; apply `gpu_augment` on device.
    train=False yields normalized float batches."""
    x, y = get_split_tensors(split_name)
    ds = XrayTensorDataset(x, y, (lambda t: t) if train else normalize_transform())
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=0, pin_memory=True)


def class_weights(y: torch.Tensor) -> torch.Tensor:
    """Inverse-frequency weights: n_total / (n_classes * n_c)."""
    counts = torch.bincount(y, minlength=len(CLASSES)).float()
    return counts.sum() / (len(CLASSES) * counts)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--src", required=True, type=Path)
    s = sub.add_parser("split")
    s.add_argument("--val-frac", type=float, default=0.15)
    s.add_argument("--seed", type=int, default=42)
    sub.add_parser("counts")
    args = ap.parse_args()
    if args.cmd == "prepare":
        prepare(args.src)
    elif args.cmd == "split":
        split(val_frac=args.val_frac, seed=args.seed)
    else:
        print_counts()


if __name__ == "__main__":
    main()
