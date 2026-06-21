# Vision Layer — Multimodal Influenza Health Monitor

## Background

This is the **Vision Layer** of a multimodal influenza health monitoring system. It processes facial video recordings to detect visual cues of respiratory illness — breathing patterns, cough events, and facial indicators — and outputs a health classification (healthy / semi-healthy / unhealthy) along with a rich feature vector for downstream fusion with other modalities (e.g., physiological signals from PPG).

### Dataset

We use the **[UBFC-rPPG](https://sites.google.com/view/ybenezeth/ubfcrppg)** dataset — 50 subjects recorded with a webcam while a pulse oximeter captures PPG (photoplethysmography) ground truth. The dataset was originally collected for remote heart-rate estimation; we adapt it for influenza monitoring by extracting breathing rate from the PPG waveform and detecting cough-like chest movements via optical flow.

| Subset | Subjects | Files | Content |
|---|---|---|---|
| UBFC1 | 8 | `vid.avi` + `gtdump.xmp` | Timestamp, HR, SpO₂, PPG waveform |
| UBFC2 | 42 | `vid.avi` + `ground_truth.txt` | PPG waveform (space-delimited floats) |

### Model

**Swin Transformer (Tiny variant)** — a hierarchical vision transformer that processes images in shifted windows.

- **Parameters**: 27.5 million
- **Input**: Single frame (224×224×3) from a 10-second video window
- **Output**: 3-class health prediction + 768-dimensional feature vector
- **Why Swin-Tiny**: 3× smaller and faster than Swin-Base; meets the <100ms deployment latency target on Intel GPU (28.6ms)

### Pipeline Architecture

```
┌──────────────┐    ┌─────────────────┐    ┌──────────────┐    ┌──────────────┐
│  UBFC Videos  │ → │  10s Windows    │ → │  Face/Chest   │ → │  Swin-Tiny   │
│  (50 subjects)│    │  (224×224)      │    │  ROI Crop     │    │  Classifier  │
└──────────────┘    └─────────────────┘    └──────────────┘    └──────────────┘
                           │                                           │
                           ▼                                           ▼
                    ┌─────────────────┐                    ┌──────────────────┐
                    │  PPG Breathing  │                    │  predictions.csv │
                    │  Rate + Cough   │                    │  + feature vector │
                    │  → Labels       │                    │  → Fusion Layer   │
                    └─────────────────┘                    └──────────────────┘
```

---

## Project structure

```
vision_layer/
├── config.py                        # Paths, hyperparameters, labels
├── dataset.py                       # Video windowing + JPEG cache
├── preprocessing.py                 # Face/chest ROI, PPG analysis, labels
├── model.py                         # Swin-Tiny classifier definition
├── train.py                         # Single-experiment training loop
├── run_experiments.py               # Batch experiment runner (5 configs)
├── evaluate.py                      # Metrics, confusion matrix, CSV export
├── export_onnx.py                   # ONNX export
├── optimize_openvino.py             # OpenVINO IR + INT8 quantization
├── requirements.txt                 # Python dependencies
│
├── output/
│   ├── predictions/
│   │   ├── predictions.csv          # Per-window model output (Fusion Layer)
│   │   ├── experiment_results_with_accuracy.csv  # Experiment summary
│   │   └── confusion_matrix.png
│   ├── checkpoints/
│   │   └── exp*_best.pt             # Trained model weights
│   ├── onnx/
│   │   └── swin_health.onnx         # ONNX export
│   ├── openvino/
│   │   └── swin_health.xml/.bin     # OpenVINO IR
│   ├── frame_cache/                 # JPEG cache for middle frames
│   └── labels.csv                   # Generated pseudo-labels
│
├── SUMMARY.md                       # This project's approach & results
└── README.md                        # This file
```

---

## Setup

### 1. Requirements

- Python 3.9+
- CUDA-capable GPU recommended (GTX 1650 or better)
- 100+ GB free disk space (for UBFC dataset ~86 GB)

### 2. Install dependencies

```bash
cd vision_layer
pip install -r requirements.txt
```

### 3. Download the dataset

Place the UBFC dataset at:
```
UBFC_rPPG/datasets/UBFC1/   (8 subject folders)
UBFC_rPPG/datasets/UBFC2/   (42 subject folders)
```

Use the included `download_ubfc.py` script in the parent directory if needed.

---

## Usage

### Quick start: generate all outputs

```bash
cd vision_layer
python run_experiments.py
```

This runs 5 Swin-Tiny experiments, saves checkpoints, generates `predictions.csv` and `experiment_results_with_accuracy.csv`, and exports ONNX + OpenVINO models.

### Individual steps

**1. Generate labels**
```bash
python preprocessing.py
```

**2. Train a single model**
```bash
python train.py --epochs 20 --batch-size 4
```

**3. Evaluate and export predictions**
```bash
python evaluate.py --split all
```

**4. Export ONNX**
```bash
python export_onnx.py
```

**5. OpenVINO optimization**
```bash
python optimize_openvino.py
```

---

## Output format

### predictions.csv (Fusion Layer input)

| Column | Type | Description |
|---|---|---|
| `filename` | str | Unique window ID, e.g., `UBFC1/5-gt/win_000000` |
| `prediction` | int | Model output: 0=healthy, 1=semi_healthy, 2=unhealthy |
| `label` | int | Ground truth (from PPG heuristics) |
| `feature_vector` | str | JSON-encoded 768-dimensional float array — concatenated by Fusion Layer |

### experiment_results_with_accuracy.csv

One row per experiment. Columns include hyperparameters (`gamma`, `epochs`, `d_model`, `batch_size`), evaluation metrics (`test_f1_macro`, `test_accuracy`), confusion matrix cells (`cm_healthy_to_healthy`, ...), and per-epoch training curves (`train_loss_curve`, `val_f1_curve`, ...).

---

## Configuration

Edit `config.py` to adjust:

| Parameter | Default | Description |
|---|---|---|
| `WINDOW_SEC` | 10.0 | Window duration |
| `STRIDE_SEC` | 5.0 | Stride between windows |
| `TARGET_FPS` | 30 | Frame rate |
| `FRAME_SIZE` | (224, 224) | Input resolution |
| `USE_YOLO_WORLD` | False | Toggle YOLO-World detection (slow) |
| `NUM_CLASSES` | 3 | Health categories |
| `FEATURE_DIM` | 768 | Swin-Tiny feature vector size |
| `BATCH_SIZE` | 4 | Training batch size |
| `LEARNING_RATE` | 1e-4 | AdamW learning rate |
| `FOCAL_GAMMA` | 2.0 | Focal loss gamma (0 = plain CE) |
| `EARLY_STOP_PATIENCE` | 5 | Epochs without improvement before stopping |

---

## Large files excluded from GitHub

Some generated files exceed GitHub's 100 MB file size limit and are excluded via `.gitignore`. These must be regenerated locally:

| File | Size | How to regenerate |
|---|---|---|
| `output/checkpoints/exp*_best.pt` | 106 MB each (×5) | `python run_experiments.py` |
| `output/onnx/swin_health.onnx` | 113 MB | `python export_onnx.py` |
| `output/openvino/swin_health.xml` / `.bin` | 152 MB | `python optimize_openvino.py` |
| `output/frame_cache/*.jpg` | ~10 MB | Auto-created on first dataset run |

The old Swin-B checkpoint (`best_model.pt`, ~994 MB) has been removed and is no longer needed.

---

## Deployment

The exported ONNX model is ready for inference with ONNX Runtime or OpenVINO:

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("swin_health.onnx")
frame = np.random.randn(1, 3, 224, 224).astype(np.float32)
logits, features = session.run(None, {"input": frame})
# logits: (1, 3) → health class
# features: (1, 768) → Fusion Layer input
```

For Intel hardware (DK-2500), use the OpenVINO IR model at `output/openvino/swin_health.xml` — benchmarks at **28.6ms** on Intel Iris Xe GPU.
