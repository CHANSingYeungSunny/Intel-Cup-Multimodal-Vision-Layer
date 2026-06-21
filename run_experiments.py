"""
Batch experiment runner — 5 Swin-Tiny configurations + Swin-B baseline.
Writes experiment_results_with_accuracy.csv and predictions.csv.
"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).parent))

from dataset import UBFCWindowDataset, _collate_windows, _list_subjects
from model import build_model, FocalLoss
from sklearn.metrics import (
    f1_score, accuracy_score, precision_recall_fscore_support, confusion_matrix,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
EARLY_STOP_PATIENCE = 5
torch.manual_seed(SEED); np.random.seed(SEED)

ROOT = Path(__file__).parent
OUT = ROOT / "output" / "predictions"
CKPT = ROOT / "output" / "checkpoints"
OUT.mkdir(parents=True, exist_ok=True); CKPT.mkdir(parents=True, exist_ok=True)

# ── Swin-Tiny experiments ──
EXPERIMENTS = [
    # Main: runs until convergence via early stopping (max 50 epochs)
    (1, "g2.0_e50_d768_l4_b4_clinical_SwinT",          2.0, 50, 4, 0, "clinical"),
    # Ablations: capped at 20 epochs each
    (2, "g2.0_e20_d768_l4_b8_clinical_SwinT",          2.0, 20, 8, 0, "clinical"),
    (3, "g2.0_e20_d768_l4_b4_clinical_SwinT",          2.0, 20, 4, 0, "clinical"),
    (4, "g2.0_e20_d768_l4_b4_frozen2_clinical_SwinT",  2.0, 20, 4, 2, "clinical"),
    (5, "g2.0_e20_d768_l4_b4_quantile_SwinT",          2.0, 20, 4, 0, "quantile"),
]

# ── load labels ──
labels_df = pd.read_csv(ROOT / "output" / "labels.csv")
label_map = dict(zip(labels_df["window_id"], labels_df["label"]))

# Quantile-based label thresholds
if "breathing_rate" in labels_df.columns:
    br = labels_df["breathing_rate"].dropna(); br = br[br > 0]
    q33, q66 = br.quantile([0.33, 0.66])
    quantile_map = {}
    for _, row in labels_df.iterrows():
        r = row["breathing_rate"]
        if r <= 0: quantile_map[row["window_id"]] = 1
        elif r <= q33: quantile_map[row["window_id"]] = 0
        elif r <= q66: quantile_map[row["window_id"]] = 1
        else: quantile_map[row["window_id"]] = 2
    print(f"Quantile thresholds: {q33:.1f}, {q66:.1f} bpm")

# ── train/val split ──
subjects = _list_subjects()
n = len(subjects); rng = np.random.default_rng(SEED)
idx = list(range(n)); rng.shuffle(idx)
split = int(n * 0.8)
train_subs = [subjects[i] for i in idx[:split]]
val_subs = [subjects[i] for i in idx[split:]]
print(f"Train: {len(train_subs)} subjects | Val: {len(val_subs)} subjects")

# Build datasets once
train_ds = UBFCWindowDataset(train_subs)
val_ds = UBFCWindowDataset(val_subs)

def make_loaders(bs):
    tl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0, collate_fn=_collate_windows)
    vl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0, collate_fn=_collate_windows)
    return tl, vl

# ── run experiments ──
all_rows = []
best_exp_f1 = -1.0
best_exp_ckpt = None
best_exp_id = None

for exp_id, cfg_label, gamma, epochs, batch_size, freeze_stages, label_mode in EXPERIMENTS:
    print(f"\n{'='*60}")
    print(f"Exp {exp_id}: {cfg_label}")
    print(f"{'='*60}")

    train_loader, val_loader = make_loaders(batch_size)
    current_label_map = quantile_map if label_mode == "quantile" else label_map

    model = build_model(freeze_stages=freeze_stages)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    alpha = torch.tensor([1.0, 1.5, 2.0], device=DEVICE)
    criterion = FocalLoss(alpha=alpha, gamma=gamma)

    best_f1 = 0.0; best_epoch = 0; best_state = None; patience_left = EARLY_STOP_PATIENCE
    train_losses, val_losses, train_f1s, val_f1s = [], [], [], []

    for epoch in range(epochs):
        # train
        model.train()
        tl, t_preds, t_labels = 0.0, [], []
        for batch in train_loader:
            frames = batch["frames"].to(DEVICE); mid = frames.size(1)//2; x = frames[:,mid,:,:,:]
            y_list = [current_label_map.get(w, 1) for w in batch["window_id"]]
            y = torch.tensor(y_list, device=DEVICE, dtype=torch.long)
            optimizer.zero_grad(); logits, _ = model(x); loss = criterion(logits, y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item() * x.size(0)
            t_preds.extend(logits.argmax(-1).cpu().numpy()); t_labels.extend(y.cpu().numpy())
        scheduler.step()
        train_losses.append(round(tl / len(train_ds), 4))
        train_f1s.append(round(f1_score(t_labels, t_preds, average="macro"), 4))

        # val
        model.eval()
        vl, v_preds, v_labels = 0.0, [], []
        with torch.no_grad():
            for batch in val_loader:
                frames = batch["frames"].to(DEVICE); mid = frames.size(1)//2; x = frames[:,mid,:,:,:]
                y_list = [current_label_map.get(w, 1) for w in batch["window_id"]]
                y = torch.tensor(y_list, device=DEVICE, dtype=torch.long)
                logits, _ = model(x); loss = criterion(logits, y)
                vl += loss.item() * x.size(0)
                v_preds.extend(logits.argmax(-1).cpu().numpy()); v_labels.extend(y.cpu().numpy())
        val_losses.append(round(vl / len(val_ds), 4))
        vf1 = round(f1_score(v_labels, v_preds, average="macro"), 4)
        val_f1s.append(vf1)
        if vf1 > best_f1:
            best_f1 = vf1; best_epoch = epoch; patience_left = EARLY_STOP_PATIENCE
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1

        print(f"  E{epoch+1:3d}: loss={train_losses[-1]:.4f}/{val_losses[-1]:.4f}  f1={train_f1s[-1]:.4f}/{vf1:.4f}  best={best_f1:.4f}@{best_epoch+1}  p={patience_left}")

        if patience_left <= 0:
            print(f"  Early stopping at epoch {epoch+1} (no improvement for {EARLY_STOP_PATIENCE} epochs)")
            break

    # save checkpoint
    ckpt_path = CKPT / f"exp{exp_id}_best.pt"
    torch.save({"epoch": best_epoch, "best_f1": best_f1, "model": best_state, "config": cfg_label}, ckpt_path)
    print(f"  Saved: {ckpt_path}")

    if best_f1 > best_exp_f1:
        best_exp_f1 = best_f1; best_exp_ckpt = ckpt_path; best_exp_id = exp_id

    # final eval
    y_true = np.array(v_labels); y_pred = np.array(v_preds)
    acc = round(accuracy_score(y_true, y_pred), 4)
    p_m, r_m, f_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    row = {
        "exp_id": exp_id, "config_label": cfg_label,
        "gamma": gamma, "epochs": epochs, "d_model": 768, "n_layers": 4,
        "batch_size": batch_size, "label_mode": label_mode,
        "best_epoch": best_epoch, "val_macro_f1_best": best_f1,
        "accuracy": acc, "test_accuracy": acc,
        "test_precision_macro": round(p_m,4), "test_recall_macro": round(r_m,4), "test_f1_macro": round(f_m,4),
        "test_precision_weighted": round(p_w,4), "test_recall_weighted": round(r_w,4), "test_f1_weighted": round(f_w,4),
        "test_loss": val_losses[-1],
        "cm_healthy_to_healthy": int(cm[0,0]), "cm_healthy_to_semi": int(cm[0,1]), "cm_healthy_to_unhealthy": int(cm[0,2]),
        "cm_semi_to_healthy": int(cm[1,0]), "cm_semi_to_semi": int(cm[1,1]), "cm_semi_to_unhealthy": int(cm[1,2]),
        "cm_unhealthy_to_healthy": int(cm[2,0]), "cm_unhealthy_to_semi": int(cm[2,1]), "cm_unhealthy_to_unhealthy": int(cm[2,2]),
        "train_loss_curve": json.dumps(train_losses), "val_loss_curve": json.dumps(val_losses),
        "train_acc_curve": json.dumps(train_f1s), "val_acc_curve": json.dumps(val_f1s), "val_f1_curve": json.dumps(val_f1s),
    }
    all_rows.append(row)

# ── Write experiment_results CSV ──
csv_path = OUT / "experiment_results_with_accuracy.csv"
df_new = pd.DataFrame(all_rows)

# Preserve Swin-B baseline if it exists, tag it
swinb_path = OUT / "experiment_results_with_accuracy.csv"
existing_swinb = None
if swinb_path.exists():
    old = pd.read_csv(swinb_path)
    # Check if old row is Swin-B (d_model=1024)
    if len(old) > 0 and old.iloc[0].get("d_model", 0) == 1024:
        old.loc[0, "config_label"] = str(old.loc[0, "config_label"]).replace("_SwinT", "") + "_SwinB"
        existing_swinb = old

if existing_swinb is not None:
    df_final = pd.concat([existing_swinb, df_new], ignore_index=True)
else:
    df_final = df_new

df_final.to_csv(csv_path, index=False)
print(f"\nSaved {len(df_final)} rows to {csv_path}")
print(df_final[["exp_id","config_label","test_f1_macro","best_epoch"]].to_string(index=False))

# ── Generate predictions.csv from best Swin-Tiny model ──
print(f"\n{'='*60}")
print(f"Generating predictions.csv from best Swin-Tiny model (exp {best_exp_id}, F1={best_exp_f1:.4f})")
print(f"{'='*60}")

best_model = build_model()
best_model.load_state_dict(torch.load(best_exp_ckpt, map_location=DEVICE)["model"])
best_model.to(DEVICE)
best_model.eval()

full_ds = UBFCWindowDataset()
full_loader = DataLoader(full_ds, batch_size=4, shuffle=False, num_workers=0, collate_fn=_collate_windows)

all_wids, all_preds, all_labels, all_feats = [], [], [], []
with torch.no_grad():
    for batch in full_loader:
        frames = batch["frames"].to(DEVICE); mid = frames.size(1)//2; x = frames[:,mid,:,:,:]
        y_list = [label_map.get(w, 1) for w in batch["window_id"]]
        logits, feats = best_model(x)
        all_preds.extend(logits.argmax(-1).cpu().numpy())
        all_labels.extend(y_list)
        all_wids.extend(batch["window_id"])
        all_feats.append(feats.cpu().numpy())

feats_arr = np.concatenate(all_feats, axis=0)
pred_rows = []
for i in range(len(all_wids)):
    pred_rows.append({
        "filename": all_wids[i],
        "prediction": int(all_preds[i]),
        "label": int(all_labels[i]),
        "feature_vector": json.dumps(feats_arr[i].tolist()),
    })
pdf = pd.DataFrame(pred_rows)
pred_path = OUT / "predictions.csv"
pdf.to_csv(pred_path, index=False)
acc = (np.array(all_preds) == np.array(all_labels)).mean()
print(f"predictions.csv: {len(pdf)} rows, accuracy={acc:.4f}")
print(f"Columns: {list(pdf.columns)}")
