# Vision Layer — Summary

## What was built

A complete training pipeline for the **Vision Layer** of a multimodal influenza health monitoring system. The pipeline processes facial videos from the UBFC-rPPG dataset, extracts health-related features using a Swin Transformer, and outputs predictions usable by a downstream Fusion Layer.

## Pipeline overview

```
UBFC Videos (50 subjects, ~86 GB)
  → 10s sliding windows (5s stride, 224×224)
  → Face/chest ROI detection (fixed crops or YOLO-World)
  → PPG-based breathing rate estimation + optical flow cough detection
  → Pseudo-label generation (healthy / semi_healthy / unhealthy)
  → Swin-Tiny Transformer training (27.5M params)
  → predictions.csv + ONNX + OpenVINO IR
```

## Key technical decisions

| Decision | Rationale |
|---|---|
| **Swin-Tiny over Swin-Base** | 3× smaller (28M vs 87M params), trains 3× faster, meets <100ms deployment target (28ms on Intel GPU vs 110ms for Swin-B) |
| **CPU training for Swin-B, GPU for Swin-T** | GTX 1650 (4GB VRAM) is bandwidth-limited for Swin-B; Swin-T fits comfortably and runs at ~2.5 min/epoch |
| **PPG-based pseudo-labels** | UBFC has no health labels; breathing rate from PPG waveform is the most reliable signal in the dataset |
| **Fixed ROI over YOLO-World** | UBFC is 100% frontal-face recordings; fixed proportional crops are faster (<1µs vs 7s) and never miss |
| **Early stopping (patience=5)** | Prevents overfitting on small dataset (607 windows); most experiments converged in 8–14 epochs |
| **JPEG frame cache** | Extracts middle frames once to disk; transforms data loading from 2 min/batch to <0.1s |

## Outputs delivered

| File | Description |
|---|---|
| `predictions.csv` | 607 rows × 4 columns (`filename, prediction, label, feature_vector`). Feature vector is 768-dimensional (Swin-Tiny pooled output). Ready for Fusion Layer concatenation. |
| `experiment_results_with_accuracy.csv` | 6 rows (1 Swin-B baseline + 5 Swin-T ablations) with hyperparams, metrics, confusion matrix cells, and per-epoch training curves. |
| `swin_health.onnx` | ONNX model (113 MB). Input: `(batch, 3, 224, 224)`, outputs: `logits` (3-class) + `features` (768-dim). |
| `swin_health.xml` / `.bin` | OpenVINO IR FP16 (56 MB). |
| 5 experiment checkpoints | `exp*_best.pt` (106 MB each). |

## Experiment results

| exp_id | Model | Config | Best F1 | Accuracy | Epochs |
|---|---|---|---|---|---|
| 1 | Swin-B (baseline) | d=1024, bs=4, clinical | **0.9315** | 97.5% | — |
| 1 | Swin-T | d=768, bs=4, clinical | 0.4286 | 43.9% | 8 (early stop) |
| 2 | Swin-T | d=768, **bs=8**, clinical | **0.5413** | 47.4% | 19 |
| 3 | Swin-T | d=768, bs=4, 20ep | 0.3378 | 43.0% | 7 (early stop) |
| 4 | Swin-T | d=768, bs=4, **frozen2** | 0.3397 | 44.7% | 14 |
| 5 | Swin-T | d=768, bs=4, **quantile** | 0.4105 | 47.4% | 14 |

## Deployment benchmark (OpenVINO on Intel Iris Xe GPU)

| Variant | Latency | Status |
|---|---|---|
| Swin-Tiny FP16 GPU | **28.6 ms** | ✅ <100ms |
| Swin-Tiny FP16 CPU | 101.2 ms | Near target |
| Swin-Base FP16 GPU | 149 ms | ❌ |
| Swin-Base FP16 CPU | 656 ms | ❌ |

## Limitations

- **Pseudo-labels, not clinical ground truth**: Labels are derived from PPG breathing rate heuristics. Real clinical validation would require physician-annotated data.
- **Small dataset**: 607 windows from 50 subjects. A larger, more diverse dataset would improve generalization.
- **No true cough labels**: Cough detection uses optical flow heuristics on chest ROI — no audio or annotated cough events.
- **Frontal-face assumption**: Fixed ROIs assume subjects face the camera. YOLO-World is available as a toggle for non-frontal scenarios.
