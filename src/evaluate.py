"""Metrics, threshold selection (VALIDATION only), and the single final test evaluation.

CLI (CMD):
    python -m src.evaluate final          # picks best run by val AUC, tests ONCE

The final step refuses to run a second time (guard file models/test_results.json),
because the test set must be evaluated exactly once.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve  # noqa: E402

from src.data import CLASSES, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, make_loader, read_manifest  # noqa: E402
from src.model import load_checkpoint, save_checkpoint  # noqa: E402

TARGET_SENSITIVITY = 0.95
EXPERIMENT = "chest-xray-pneumonia"
MODELS_DIR = Path("models")
GUARD = MODELS_DIR / "test_results.json"


def tracking_uri() -> str:
    return Path("mlruns").resolve().as_uri()


@torch.no_grad()
def predict_probs(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x.to(device, non_blocking=True))
        probs.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def choose_threshold(y: np.ndarray, p: np.ndarray, target_sens: float = TARGET_SENSITIVITY) -> float:
    """Highest threshold whose sensitivity (TPR) is >= target. Use on VALIDATION only."""
    fpr, tpr, thr = roc_curve(y, p)
    idx = int(np.argmax(tpr >= target_sens))  # first (= highest threshold) index meeting target
    return float(min(thr[idx], 1.0))


def metrics_at(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y, p)),
        "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "ppv": tp / (tp + fp) if tp + fp else float("nan"),
        "npv": tn / (tn + fn) if tn + fn else float("nan"),
        "accuracy": (tp + tn) / len(y),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def plot_confusion(m: dict, title: str, path: Path) -> Path:
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(
            j,
            i,
            str(v),
            ha="center",
            va="center",
            fontsize=14,
            color="white" if v > cm.max() / 2 else "black",
        )
    ax.set_xticks([0, 1], [f"pred {c}" for c in CLASSES])
    ax.set_yticks([0, 1], [f"true {c}" for c in CLASSES])
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_roc(y: np.ndarray, p: np.ndarray, threshold: float, title: str, path: Path) -> Path:
    fpr, tpr, _ = roc_curve(y, p)
    m = metrics_at(y, p, threshold)
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {m['auc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.scatter(
        [1 - m["specificity"]],
        [m["sensitivity"]],
        color="red",
        zorder=3,
        label=f"t={threshold:.3f} (sens {m['sensitivity']:.2f}, spec {m['specificity']:.2f})",
    )
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _best_run() -> mlflow.entities.Run:
    client = mlflow.MlflowClient(tracking_uri())
    exp = client.get_experiment_by_name(EXPERIMENT)
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string="attributes.status = 'FINISHED' and tags.stage = 'train'",
        order_by=["metrics.best_val_auc DESC"],
    )
    if not runs:
        raise RuntimeError("No finished training runs found.")
    print("Candidate runs (ranked by VALIDATION AUC):")
    for r in runs:
        print(f"  {r.data.tags.get('mlflow.runName'):<28} val_auc={r.data.metrics['best_val_auc']:.6f}")
    return runs[0]


def final_evaluation() -> None:
    if GUARD.exists():
        raise SystemExit(
            f"{GUARD} exists: the test set has already been evaluated once. "
            "Re-evaluating would violate the pre-registered protocol."
        )
    mlflow.set_tracking_uri(tracking_uri())
    run = _best_run()
    run_name = run.data.tags["mlflow.runName"]
    ckpt_path = MODELS_DIR / "candidates" / f"{run_name}.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_checkpoint(ckpt_path)
    model.to(device)

    # 1) Threshold from VALIDATION only
    pv, yv = predict_probs(model, make_loader("val", 64), device)
    threshold = choose_threshold(yv, pv)
    val_m = metrics_at(yv, pv, threshold)

    # 2) Test set, ONCE, with the frozen threshold
    pt, yt = predict_probs(model, make_loader("test", 64), device)
    test_m = metrics_at(yt, pt, threshold)

    criteria = {
        "test_auc_ge_0.90": bool(test_m["auc"] >= 0.90),
        "test_sensitivity_ge_0.95": bool(test_m["sensitivity"] >= 0.95),
        "threshold_chosen_on_validation_only": True,
        "test_evaluated_once": True,
    }

    MODELS_DIR.mkdir(exist_ok=True)
    save_checkpoint(
        model.cpu(),
        meta["arch"],
        MODELS_DIR / "model.pt",
        threshold=threshold,
        class_names=list(CLASSES),
        img_size=IMG_SIZE,
    )

    rows = read_manifest()
    split_counts = {
        s: {c: sum(1 for r in rows if r["split"] == s and r["class"] == c) for c in CLASSES}
        for s in ("train", "val", "test")
    }
    card = {
        "model_name": "chest-xray-pneumonia-classifier",
        "disclaimer": "Educational demo only. Not a medical device. Not for clinical use.",
        "architecture": meta["arch"],
        "pretrained_on": "ImageNet-1k (torchvision IMAGENET1K_V1)",
        "classes": list(CLASSES),
        "positive_class": "PNEUMONIA",
        "input": {
            "size": [IMG_SIZE, IMG_SIZE],
            "channels": "grayscale replicated to RGB",
            "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        },
        "threshold": threshold,
        "threshold_selection": f"Highest threshold with validation sensitivity >= {TARGET_SENSITIVITY}",
        "validation_metrics": val_m,
        "test_metrics": test_m,
        "pre_registered_criteria": criteria,
        "training": {k: v for k, v in run.data.params.items()},
        "mlflow_run_id": run.info.run_id,
        "mlflow_run_name": run_name,
        "dataset": {
            "name": "Chest X-Ray Images (Pneumonia)",
            "source": "Kermany et al., Mendeley Data v2, doi:10.17632/rscbjbr9sj.2 "
            "(Kaggle mirror paultimothymooney/chest-xray-pneumonia)",
            "license": "CC BY 4.0",
            "population": "Pediatric patients aged 1-5, Guangzhou Women and Children's Medical Center",
            "splits": split_counts,
            "split_method": "Patient-level 15% validation carved from official train (seed 42); "
            "official test set held out and evaluated once.",
        },
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (MODELS_DIR / "model_card.json").write_text(json.dumps(card, indent=2))
    GUARD.write_text(
        json.dumps(
            {
                "threshold": threshold,
                "test": test_m,
                "criteria": criteria,
                "run_id": run.info.run_id,
                "created_utc": card["created_utc"],
            },
            indent=2,
        )
    )

    # Log to the selected MLflow run
    art = MODELS_DIR / "_eval_artifacts"
    art.mkdir(exist_ok=True)
    with mlflow.start_run(run_id=run.info.run_id):
        mlflow.log_metric("final_threshold", threshold)
        mlflow.log_metrics({f"test_{k}": float(v) for k, v in test_m.items()})
        mlflow.set_tag("selected_for_test", "true")
        mlflow.log_artifact(
            str(
                plot_confusion(
                    test_m, f"TEST confusion matrix (t={threshold:.3f})", art / "test_confusion_matrix.png"
                )
            ),
            "test",
        )
        mlflow.log_artifact(
            str(plot_roc(yt, pt, threshold, "TEST ROC curve", art / "test_roc_curve.png")), "test"
        )
        mlflow.log_artifact(str(MODELS_DIR / "model_card.json"), "test")

    print(
        json.dumps(
            {"run": run_name, "threshold": threshold, "val": val_m, "test": test_m, "criteria": criteria},
            indent=2,
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["final"])
    ap.parse_args()
    final_evaluation()


if __name__ == "__main__":
    main()
