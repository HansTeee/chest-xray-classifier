"""Training entry point. Every run logs to MLflow (./mlruns).

Model selection within a run (best epoch) uses VALIDATION AUC only.
The test set is never loaded here.

CLI (CMD):
    python -m src.train --run-name densenet_lr1e-4_full --lr 1e-4 --unfreeze all
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from src.data import IMG_SIZE, class_weights, get_split_tensors, gpu_augment, make_loader
from src.evaluate import (
    EXPERIMENT,
    TARGET_SENSITIVITY,
    choose_threshold,
    metrics_at,
    plot_confusion,
    plot_roc,
    predict_probs,
    tracking_uri,
)
from src.model import build_model, save_checkpoint, set_trainable

CANDIDATES = Path("models/candidates")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--arch", default="densenet121")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--unfreeze", default="all", help="'all' or number of last backbone blocks to train")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-batches", type=int, default=0, help="timing/debug only; 0 = full epoch")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")

    _, y_train = get_split_tensors("train")
    train_loader = make_loader("train", args.batch_size, train=True)
    val_loader = make_loader("val", 64)

    model = build_model(args.arch, pretrained=True)
    unfreeze = None if args.unfreeze == "all" else int(args.unfreeze)
    set_trainable(model, args.arch, unfreeze)
    model.to(device)

    weights = class_weights(y_train).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.set_tag("stage", "train")
        mlflow.log_params({
            "arch": args.arch, "pretrained": "imagenet", "lr": args.lr,
            "weight_decay": args.weight_decay, "epochs": args.epochs,
            "batch_size": args.batch_size, "unfreeze": args.unfreeze,
            "trainable_params": sum(p.numel() for p in params),
            "optimizer": "AdamW", "scheduler": "cosine", "amp": device.type == "cuda",
            "img_size": IMG_SIZE, "seed": args.seed,
            "class_weights": [round(float(w), 4) for w in weights.cpu()],
            "augmentation": "rotation10, affine(translate .05, scale .95-1.05), "
                            "brightness/contrast .2, NO hflip",
            "n_train": len(y_train), "n_val": len(val_loader.dataset),
            "device": str(device),
        })

        best_auc, best_state, best_epoch = -1.0, None, -1
        for epoch in range(1, args.epochs + 1):
            model.train()
            t0 = time.perf_counter()
            losses, tr_p, tr_y = [], [], []
            for i, (x, y) in enumerate(train_loader):
                if args.max_batches and i >= args.max_batches:
                    break
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                x = gpu_augment(x)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=device.type == "cuda"):
                    logits = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                losses.append(loss.item())
                tr_p.append(torch.softmax(logits.detach().float(), 1)[:, 1].cpu())
                tr_y.append(y.cpu())
            scheduler.step()
            epoch_time = time.perf_counter() - t0

            tr_p, tr_y = torch.cat(tr_p).numpy(), torch.cat(tr_y).numpy()
            pv, yv = predict_probs(model, val_loader, device)
            with torch.no_grad():
                val_loss = criterion(
                    torch.log(torch.tensor(np.stack([1 - pv, pv], 1)).clamp_min(1e-7)).to(device),
                    torch.tensor(yv).to(device),
                ).item()
            val_auc = roc_auc_score(yv, pv)
            train_auc = roc_auc_score(tr_y, tr_p) if len(set(tr_y)) > 1 else float("nan")
            mlflow.log_metrics({
                "train_loss": float(np.mean(losses)), "val_loss": val_loss,
                "train_auc": train_auc, "val_auc": val_auc,
                "lr": scheduler.get_last_lr()[0], "epoch_time_s": epoch_time,
            }, step=epoch)
            print(f"[{args.run_name}] epoch {epoch}/{args.epochs}  train_loss {np.mean(losses):.4f}  "
                  f"val_loss {val_loss:.4f}  train_auc {train_auc:.4f}  val_auc {val_auc:.4f}  "
                  f"({epoch_time:.1f}s)", flush=True)
            if val_auc > best_auc:
                best_auc, best_epoch = val_auc, epoch
                best_state = copy.deepcopy(model.state_dict())

        # Restore best epoch (selected on VALIDATION AUC)
        model.load_state_dict(best_state)
        pv, yv = predict_probs(model, val_loader, device)
        thr = choose_threshold(yv, pv, TARGET_SENSITIVITY)
        vm = metrics_at(yv, pv, thr)
        mlflow.log_metrics({"best_val_auc": best_auc, "best_epoch": best_epoch,
                            "val_threshold": thr, "val_sensitivity": vm["sensitivity"],
                            "val_specificity": vm["specificity"], "val_accuracy": vm["accuracy"]})

        art = Path("models/_artifacts") / args.run_name
        art.mkdir(parents=True, exist_ok=True)
        mlflow.log_artifact(str(plot_confusion(vm, f"{args.run_name}: VAL confusion (t={thr:.3f})",
                                               art / "val_confusion_matrix.png")), "val")
        mlflow.log_artifact(str(plot_roc(yv, pv, thr, f"{args.run_name}: VAL ROC",
                                         art / "val_roc_curve.png")), "val")

        model.cpu().eval()
        mlflow.pytorch.log_model(model, artifact_path="model")
        CANDIDATES.mkdir(parents=True, exist_ok=True)
        save_checkpoint(model, args.arch, CANDIDATES / f"{args.run_name}.pt",
                        mlflow_run_id=run.info.run_id, best_epoch=best_epoch)
        print(f"[{args.run_name}] best val AUC {best_auc:.4f} @ epoch {best_epoch}; "
              f"val threshold {thr:.4f} -> sens {vm['sensitivity']:.3f}, spec {vm['specificity']:.3f}")


if __name__ == "__main__":
    main()
