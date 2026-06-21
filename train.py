"""
Training Loop
=============
Trains the Vision Layer Swin Transformer with WeightedRandomSampler,
Focal Loss, AdamW, and cosine-annealing scheduler.  Saves the best
checkpoint by macro F1 on the validation set.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    ADAM_BETA1,
    ADAM_BETA2,
    NUM_WORKERS,
    VAL_SPLIT,
    EARLY_STOP_PATIENCE,
    CLASS_WEIGHTS,
    FOCAL_GAMMA,
    NUM_CLASSES,
    CLASS_NAMES,
    CHECKPOINT_DIR,
    LOG_DIR,
    DEVICE,
    RANDOM_SEED,
    WARMUP_EPOCHS,
    T_0,
    T_MULT,
    OUTPUT_DIR,
)
from dataset import UBFCWindowDataset, create_dataloaders
from model import build_model, FocalLoss
from sklearn.metrics import f1_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_metrics(
    all_labels: np.ndarray,
    all_preds: np.ndarray,
) -> dict:
    """Compute accuracy and per-class + macro F1."""
    acc = float((all_labels == all_preds).mean())
    macro_f1 = float(f1_score(all_labels, all_preds, average="macro"))
    per_class_f1 = f1_score(all_labels, all_preds, average=None, labels=list(range(NUM_CLASSES)))
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_f1": {CLASS_NAMES[i]: float(f) for i, f in enumerate(per_class_f1)},
    }


def _build_sampler(dataset: UBFCWindowDataset, labels_df: "pd.DataFrame") -> WeightedRandomSampler:
    """Create a WeightedRandomSampler from per-window labels."""
    import pandas as pd
    label_map = dict(zip(labels_df["window_id"], labels_df["label"]))
    sample_labels = []
    for i in range(len(dataset)):
        wid = dataset[i]["window_id"]
        sample_labels.append(label_map.get(wid, 0))

    labels_arr = np.array(sample_labels)
    class_counts = np.bincount(labels_arr, minlength=NUM_CLASSES)
    class_weights_arr = 1.0 / (class_counts + 1)
    sample_weights = class_weights_arr[labels_arr]

    print(f"Class distribution: {dict(zip(CLASS_NAMES.values(), class_counts))}")
    print(f"Sample weights (mean): {sample_weights.mean():.4f}")

    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(sample_weights),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    model: nn.Module | None = None,
    train_loader: DataLoader | None = None,
    val_loader: DataLoader | None = None,
    labels_csv: Path | None = None,
    resume_from: Path | None = None,
) -> Path:
    """
    Run the full training loop.

    Returns the path to the best checkpoint.
    """
    import pandas as pd

    # ---- Setup ----
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if model is None:
        model = build_model()

    if train_loader is None or val_loader is None:
        train_loader, val_loader = create_dataloaders(
            batch_size=BATCH_SIZE,
            val_split=VAL_SPLIT,
        )

    # Preprocess labels if not yet cached
    if labels_csv is None:
        labels_csv = OUTPUT_DIR / "labels.csv"
    if not labels_csv.exists():
        print("Label cache not found. Run preprocessing.preprocess_and_cache_labels() first.")
        print("Generating labels now (this will take a while)...")
        from preprocessing import preprocess_and_cache_labels
        # Build a full dataset for label gen
        full_ds = UBFCWindowDataset()
        labels_csv = preprocess_and_cache_labels(full_ds)

    labels_df = pd.read_csv(labels_csv)

    # Weighted sampler for training
    train_ds = train_loader.dataset
    sampler = _build_sampler(train_ds, labels_df)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=train_loader.collate_fn,
    )

    # ---- Optimiser & Scheduler ----
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(ADAM_BETA1, ADAM_BETA2),
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_MULT
    )

    # ---- Loss ----
    class_weights_tensor = torch.tensor(CLASS_WEIGHTS, device=DEVICE, dtype=torch.float32)
    criterion = FocalLoss(alpha=class_weights_tensor, gamma=FOCAL_GAMMA)

    # ---- Resume ----
    start_epoch = 0
    best_f1 = 0.0
    best_path: Path | None = None
    patience_counter = 0

    if resume_from is not None and resume_from.exists():
        ckpt = torch.load(resume_from, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_f1 = ckpt.get("best_f1", 0.0)
        print(f"Resumed from {resume_from} (epoch {start_epoch})")

    # ---- TensorBoard ----
    writer = SummaryWriter(log_dir=str(LOG_DIR))

    # ---- Training Loop ----
    model.train()
    for epoch in range(start_epoch, NUM_EPOCHS):
        t0 = time.time()
        total_loss = 0.0
        all_labels: list[int] = []
        all_preds: list[int] = []

        # --- Train ---
        model.train()
        label_map = dict(zip(labels_df["window_id"], labels_df["label"]))
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [train]")
        for batch in pbar:
            frames = batch["frames"].to(DEVICE)          # (B, T, C, H, W)
            B = frames.size(0)

            # Use middle frame for Swin (classification on a single frame)
            mid = frames.size(1) // 2
            x = frames[:, mid, :, :, :]                   # (B, 3, H, W)

            # Map labels from window IDs
            labels_list = [label_map.get(wid, 0) for wid in batch["window_id"]]
            y = torch.tensor(labels_list, device=DEVICE, dtype=torch.long)

            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * B
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.cpu().numpy().tolist())

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_loss / len(train_ds)
        train_metrics = _compute_metrics(np.array(all_labels), np.array(all_preds))
        scheduler.step()

        # --- Validation ---
        val_loss, val_labels, val_preds = _validate(model, val_loader, criterion, labels_df)
        val_metrics = _compute_metrics(np.array(val_labels), np.array(val_preds))

        elapsed = time.time() - t0
        lr = scheduler.get_last_lr()[0]

        # --- Logging ---
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("F1/train", train_metrics["macro_f1"], epoch)
        writer.add_scalar("F1/val", val_metrics["macro_f1"], epoch)
        writer.add_scalar("LR", lr, epoch)

        print(
            f"Epoch {epoch+1:3d} | "
            f"Train Loss {avg_train_loss:.4f}  F1 {train_metrics['macro_f1']:.3f} | "
            f"Val Loss {val_loss:.4f}  F1 {val_metrics['macro_f1']:.3f} | "
            f"LR {lr:.2e}  ({elapsed:.0f}s)"
        )
        print(f"  Per-class Val F1: {val_metrics['per_class_f1']}")

        # --- Checkpoint ---
        val_f1 = val_metrics["macro_f1"]
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            best_path = CHECKPOINT_DIR / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_f1": best_f1,
                "class_names": CLASS_NAMES,
            }, best_path)
            print(f"  -> Saved best model (F1={best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"Early stopping after {EARLY_STOP_PATIENCE} epochs without improvement.")
                break

        # Save latest
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_f1": best_f1,
        }, CHECKPOINT_DIR / "last_model.pt")

    writer.close()

    if best_path is None:
        best_path = CHECKPOINT_DIR / "last_model.pt"

    print(f"\nTraining complete. Best F1: {best_f1:.4f}  →  {best_path}")
    return best_path


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    labels_df: "pd.DataFrame",
) -> tuple[float, list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    all_labels: list[int] = []
    all_preds: list[int] = []
    label_map = dict(zip(labels_df["window_id"], labels_df["label"]))

    for batch in loader:
        frames = batch["frames"].to(DEVICE)
        mid = frames.size(1) // 2
        x = frames[:, mid, :, :, :]

        labels_list = [label_map.get(wid, 0) for wid in batch["window_id"]]
        y = torch.tensor(labels_list, device=DEVICE, dtype=torch.long)

        logits, _ = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * frames.size(0)

        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(y.cpu().numpy().tolist())

    return total_loss / len(loader.dataset), all_labels, all_preds


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Vision Layer Swin Transformer")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--freeze-stages", type=int, default=0,
                        help="Freeze first N Swin stages")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--labels", type=str, default=None,
                        help="Path to precomputed labels.csv")
    args = parser.parse_args()

    # Override config with CLI args
    import config
    config.NUM_EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.lr

    model = build_model(freeze_stages=args.freeze_stages)
    train_loader, val_loader = create_dataloaders(batch_size=args.batch_size)
    labels_path = Path(args.labels) if args.labels else None
    resume_path = Path(args.resume) if args.resume else None

    best_ckpt = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        labels_csv=labels_path,
        resume_from=resume_path,
    )
    print(f"Best checkpoint: {best_ckpt}")
