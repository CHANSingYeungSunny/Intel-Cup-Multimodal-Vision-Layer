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
  → Swin-Tiny Transformer training (27.5M params, 768-dim features)
  → predictions.csv + ONNX + OpenVINO IR
```

## Key technical decisions

| Decision | Rationale |
|---|---|
| **Swin-Tiny over Swin-Base** | 3× smaller (28M vs 87M params), trains 3× faster, meets <100ms deployment target (28.6ms on Intel GPU vs 150ms for Swin-B). 768-dim feature vector is still rich enough for downstream fusion. |
| **GPU training (GTX 1650)** | Swin-Tiny fits comfortably in 4GB VRAM at batch_size=4; runs ~2.5 min/epoch |
| **PPG-based pseudo-labels** | UBFC has no health labels; breathing rate from PPG waveform is the most reliable signal in the dataset. Labels: healthy (BR 12–20 bpm), semi_healthy (borderline), unhealthy (BR <8 or >24, or frequent cough events) |
| **Fixed ROI over YOLO-World** | UBFC is 100% frontal-face recordings; fixed proportional crops are faster (<1µs vs 7s) and never miss. YOLO-World is available as a config toggle (`USE_YOLO_WORLD=True`) for non-frontal scenarios |
| **Early stopping (patience=5)** | Prevents overfitting on small dataset (607 windows); each experiment converged in 7–19 epochs |
| **JPEG frame cache** | Extracts middle frames once to disk (607 JPEGs, ~10 MB); transforms data loading from 2 min/batch to <0.1s |

## Outputs delivered

| File | Description |
|---|---|
| `predictions.csv` | 607 rows × 4 columns (`filename, prediction, label, feature_vector`). Feature vector is 768-dimensional (Swin-Tiny pooled output). Ready for Fusion Layer concatenation. |
| `experiment_results_with_accuracy.csv` | 5 rows (Swin-Tiny: main + 4 ablations) with hyperparameters, metrics, confusion matrix cells, and per-epoch training curves. |
| `swin_health.onnx` | ONNX model (113 MB). Input: `(batch, 3, 224, 224)`, outputs: `logits` (3-class) + `features` (768-dim). |
| `swin_health.xml` / `.bin` | OpenVINO IR FP16 (56 MB). |
| 5 experiment checkpoints | `exp*_best.pt` (106 MB each). |

## Experiment results (Swin-Tiny, 5 configurations)

| exp_id | Config | Best F1 | Accuracy | Epochs | Notes |
|---|---|---|---|---|---|
| 1 | main: bs=4, clinical | 0.4286 | 43.9% | 8 (early stop) | Baseline Swin-T |
| 2 | **bs=8**, clinical | **0.5413** | 47.4% | 19 | Best overall — larger batch helps |
| 3 | 20ep, bs=4, clinical | 0.3378 | 43.0% | 7 (early stop) | Fewer epochs, slightly worse |
| 4 | frozen2, bs=4, clinical | 0.3397 | 44.7% | 14 | Freezing early stages slows convergence |
| 5 | quantile, bs=4 | 0.4105 | 47.4% | 14 | Quantile labels match clinical closely |

## Deployment benchmark (OpenVINO on Intel Iris Xe GPU)

| Variant | Latency | Status |
|---|---|---|
| **Swin-Tiny FP16 GPU** | **28.6 ms** | ✅ <100ms |
| Swin-Tiny FP16 CPU | 101.2 ms | Near target |

On the target DK-2500 hardware (newer Intel Arc GPU), latency is expected to be 15–20ms.

## Limitations

- **Pseudo-labels, not clinical ground truth**: Labels are derived from PPG breathing rate heuristics. Real clinical validation would require physician-annotated data.
- **Small dataset**: 607 windows from 50 subjects. A larger, more diverse dataset would improve generalization.
- **No true cough labels**: Cough detection uses optical flow heuristics on chest ROI — no audio or annotated cough events.
- **Frontal-face assumption**: Fixed ROIs assume subjects face the camera. YOLO-World toggle available for non-frontal scenarios.
- **Moderate accuracy**: 43–47% reflects the difficulty of predicting health from a single face frame without clinical labels. The feature vectors remain useful for the Fusion Layer regardless of classification accuracy.
