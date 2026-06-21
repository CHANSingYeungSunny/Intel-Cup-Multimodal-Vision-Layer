"""
Evaluation
==========
Run the trained Swin Transformer on the validation set (or full dataset)
and produce:
  - Accuracy, Precision, Recall, F1-score (per-class + macro).
  - Classification report (sklearn).
  - Confusion matrix (matplotlib).
  - predictions.csv  with columns:  filename, prediction, label, feature_vector
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    NUM_CLASSES,
    CLASS_NAMES,
    NUM_WORKERS,
    PREDICTION_DIR,
    DEVICE,
    CHECKPOINT_DIR,
    FEATURE_DIM,
)
from dataset import create_dataloaders, _collate_windows, UBFCWindowDataset
from model import build_model


# ---------------------------------------------------------------------------
# Experiment results CSV (groupmate-compatible wide format)
# ---------------------------------------------------------------------------

def _write_experiment_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cm: np.ndarray,
    output_dir: Path,
) -> None:
    """
    Write experiment_results_with_accuracy.csv in the wide format used by
    the rest of the team.  One row per run, hyperparams + metrics + curves.
    """
    import json as _json
    from config import (
        FOCAL_GAMMA, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE,
        SWIN_MODEL, NUM_CLASSES, FEATURE_DIM, CLASS_NAMES,
    )

    # Build config label (matching groupmate convention)
    # Format: g{gamma}_e{epochs}_d{d_model}_l{n_layers}_b{batch}_{label_mode}
    n_layers = 4  # Swin-B has 4 stages
    d_model = FEATURE_DIM  # 1024
    label_mode = "clinical"  # PPG breathing-rate based labels
    config_label = (
        f"g{FOCAL_GAMMA}_e{NUM_EPOCHS}_d{d_model}"
        f"_l{n_layers}_b{BATCH_SIZE}_{label_mode}"
    )

    # Per-class metrics
    p_class, r_class, f_class, s_class = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(NUM_CLASSES)), zero_division=0
    )

    # Overall metrics
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Test loss — compute once more (simplified: CE loss)
    from sklearn.metrics import log_loss
    try:
        # Use a one-hot style approach
        test_loss = float(log_loss(y_true, np.eye(NUM_CLASSES)[y_pred], labels=list(range(NUM_CLASSES))))
    except Exception:
        test_loss = 0.0

    # Flatten confusion matrix: groupmate uses 9 named columns
    cm_flat = {}
    for i, true_name in enumerate(CLASS_NAMES.values()):
        for j, pred_name in enumerate(CLASS_NAMES.values()):
            col = f"cm_{true_name}_to_{pred_name}"
            cm_flat[col] = int(cm[i, j])

    # Training curves — read from TensorBoard if available, else empty
    train_loss = _read_tb_scalars("Loss/train")
    val_loss = _read_tb_scalars("Loss/val")
    train_acc = _read_tb_scalars("F1/train")
    val_acc = _read_tb_scalars("F1/val")
    val_f1 = _read_tb_scalars("F1/val")  # same as val_acc for F1 monitoring

    # Build the single-row DataFrame
    row = {
        "exp_id": 1,
        "config_label": config_label,
        "gamma": FOCAL_GAMMA,
        "epochs": NUM_EPOCHS,
        "d_model": d_model,
        "n_layers": n_layers,
        "batch_size": BATCH_SIZE,
        "label_mode": label_mode,
        "best_epoch": 0,   # populated by train.py if available
        "val_macro_f1_best": float(macro_f),
        "accuracy": float(acc),
        "test_accuracy": float(acc),
        "test_precision_macro": float(macro_p),
        "test_recall_macro": float(macro_r),
        "test_f1_macro": float(macro_f),
        "test_precision_weighted": float(weighted_p),
        "test_recall_weighted": float(weighted_r),
        "test_f1_weighted": float(weighted_f),
        "test_loss": test_loss,
        **cm_flat,
        "train_loss_curve": _json.dumps(train_loss),
        "val_loss_curve": _json.dumps(val_loss),
        "train_acc_curve": _json.dumps(train_acc),
        "val_acc_curve": _json.dumps(val_acc),
        "val_f1_curve": _json.dumps(val_f1),
    }

    csv_path = output_dir / "experiment_results_with_accuracy.csv"
    pd.DataFrame([row]).to_csv(csv_path, index=False)
    print(f"Experiment results saved to {csv_path}")


def _read_tb_scalars(tag: str) -> list[float]:
    """Read a scalar tag from TensorBoard event logs. Returns [] if not found."""
    try:
        from pathlib import Path as _Path
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        log_dir = _Path(__file__).parent / "output" / "logs"
        if not log_dir.exists():
            return []
        for d in sorted(log_dir.iterdir()):
            if d.is_dir():
                ea = EventAccumulator(str(d))
                ea.Reload()
                if tag in ea.Tags().get("scalars", {}):
                    return [float(e.value) for e in ea.Scalars(tag)]
        return []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: "SwinHealthClassifier",
    loader: DataLoader,
    labels_df: pd.DataFrame,
    output_csv: Path | None = None,
) -> dict:
    """
    Evaluate *model* on *loader*, using *labels_df* for ground truth.

    Returns a dict of aggregated metrics.
    """
    model.eval()
    label_map = dict(zip(labels_df["window_id"], labels_df["label"]))

    all_logits: list[np.ndarray] = []
    all_feats: list[np.ndarray] = []
    all_preds: list[int] = []
    all_labels: list[int] = []
    all_wids: list[str] = []

    for batch in tqdm(loader, desc="Evaluating"):
        frames = batch["frames"].to(DEVICE)
        mid = frames.size(1) // 2
        x = frames[:, mid, :, :, :]

        labels_list = [label_map.get(wid, 0) for wid in batch["window_id"]]
        y = torch.tensor(labels_list, device=DEVICE, dtype=torch.long)

        logits, feats = model(x)
        preds = logits.argmax(dim=-1)

        all_logits.append(logits.cpu().numpy())
        all_feats.append(feats.cpu().numpy())
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())
        all_wids.extend(batch["window_id"])

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    features = np.concatenate(all_feats, axis=0)

    # ---- Metrics ----
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )

    print("=" * 60)
    print(" EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Accuracy       : {acc:.4f}")
    print(f"  Macro F1       : {macro_f1:.4f}")
    print(f"  Weighted F1    : {weighted_f1:.4f}")
    print(f"  Macro Precision: {p_macro:.4f}")
    print(f"  Macro Recall   : {r_macro:.4f}")
    print()

    per_class = {}
    for i in range(NUM_CLASSES):
        mask = y_true == i
        if mask.sum() > 0:
            p, r, f, _ = precision_recall_fscore_support(
                y_true == i, y_pred == i, average="binary"
            )
        else:
            p = r = f = 0.0
        name = CLASS_NAMES[i]
        per_class[name] = {"precision": p, "recall": r, "f1": f, "support": int(mask.sum())}
        print(f"  {name:16s}  P={p:.3f}  R={r:.3f}  F1={f:.3f}  N={int(mask.sum())}")

    print()
    print(classification_report(
        y_true, y_pred,
        target_names=[CLASS_NAMES[i] for i in range(NUM_CLASSES)],
        zero_division=0,
    ))

    # ---- Confusion Matrix ----
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[CLASS_NAMES[i] for i in range(NUM_CLASSES)],
        yticklabels=[CLASS_NAMES[i] for i in range(NUM_CLASSES)],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Vision Layer")
    fig.tight_layout()
    cm_path = PREDICTION_DIR / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to {cm_path}")

    # ---- CSV Output ----
    if output_csv is None:
        output_csv = PREDICTION_DIR / "predictions.csv"

    rows = []
    for i in range(len(y_true)):
        # Convert feature vector to compact string (JSON array)
        feat_str = json.dumps(features[i].tolist())
        rows.append({
            "filename": all_wids[i],
            "prediction": int(y_pred[i]),
            "label": int(y_true[i]),
            "feature_vector": feat_str,
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}  ({len(df_out)} rows)")

    # ---- Experiment results CSV (groupmate-compatible wide format) ----
    _write_experiment_results(y_true, y_pred, cm, PREDICTION_DIR)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Vision Layer model")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint.pt (default: best_model.pt)")
    parser.add_argument("--labels", type=str, default=None,
                        help="Path to labels.csv")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to output predictions.csv")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val", "all"],
                        help="Which split to evaluate")
    args = parser.parse_args()

    # Load model
    ckpt_path = Path(args.checkpoint) if args.checkpoint else CHECKPOINT_DIR / "best_model.pt"
    if not ckpt_path.exists():
        ckpt_path = CHECKPOINT_DIR / "last_model.pt"
    print(f"Loading checkpoint: {ckpt_path}")

    model = build_model()
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.to(DEVICE)

    # Load labels
    labels_path = Path(args.labels) if args.labels else Path(__file__).parent / "output" / "labels.csv"
    if not labels_path.exists():
        print(f"Labels not found at {labels_path}. Generating...")
        from preprocessing import preprocess_and_cache_labels
        ds = UBFCWindowDataset()
        labels_path = preprocess_and_cache_labels(ds)
    labels_df = pd.read_csv(labels_path)

    # Build dataloader for the requested split
    if args.split == "all":
        ds = UBFCWindowDataset()
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=NUM_WORKERS, collate_fn=_collate_windows)
    else:
        train_loader, val_loader = create_dataloaders(batch_size=args.batch_size)
        loader = val_loader if args.split == "val" else train_loader

    output_csv = Path(args.output) if args.output else PREDICTION_DIR / "predictions.csv"
    evaluate(model, loader, labels_df, output_csv=output_csv)
