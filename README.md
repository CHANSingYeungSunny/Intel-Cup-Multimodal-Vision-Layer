# Vision Layer — Multimodal Influenza Health Monitor

## Background

This is the **Vision Layer** of a multimodal influenza health monitoring system. It processes facial video recordings to extract **visual characteristics of respiratory illness** — face and chest regions via YOLO-World detection, cough-like chest movements via optical flow, and facial appearance features via a Swin Transformer. These visual features are combined with PPG-derived breathing rate to generate health labels, and the model outputs a classification (healthy / semi-healthy / unhealthy) along with a 768-dimensional feature vector for downstream fusion with other modalities (e.g., physiological signals from PPG).

### Dataset

We use the **[UBFC-rPPG](https://sites.google.com/view/ybenezeth/ubfcrppg)** dataset — 50 subjects recorded with a webcam while a pulse oximeter captures PPG (photoplethysmography) ground truth. The dataset was originally collected for remote heart-rate estimation; we adapt it for influenza monitoring through a combination of **visual feature extraction** and physiological signal processing:

- **YOLO-World ROI detection**: Identifies face and chest regions in each video frame. Face crops are fed into the Swin Transformer; chest crops are used for cough detection.
- **Optical flow cough detection**: Measures chest movement magnitude across frames to detect cough-like events — a visual cue of respiratory illness.
- **PPG breathing rate estimation**: Extracts breathing frequency from the PPG waveform to assess respiratory health.

These visual and physiological signals are combined to generate pseudo-labels (healthy / semi_healthy / unhealthy) for training.

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
┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐    ┌──────────────┐
│  UBFC Videos  │ → │  10s Windows    │ → │  Visual Features │ → │  Swin-Tiny   │
│  (50 subjects)│    │  (224×224)      │    │  YOLO-World ROI  │    │  Classifier  │
└──────────────┘    └─────────────────┘    │  + Optical Flow  │    └──────────────┘
                                           │  (face/chest/    │           │
                                           │   cough detection)│           ▼
                                           └──────────────────┘    ┌──────────────────┐
                                                    │              │  predictions.csv │
                                                    ▼              │  + feature vector │
                                           ┌──────────────────┐    │  → Fusion Layer   │
                                           │  PPG Breathing   │    └──────────────────┘
                                           │  Rate → Labels   │
                                           └──────────────────┘
```

**Visual characteristics extracted by the Vision Layer:**
1. **Face region** (YOLO-World / fixed ROI) — facial appearance features processed by Swin-Tiny
2. **Chest region** (YOLO-World / fixed ROI) — upper-body movement analyzed via optical flow
3. **Cough-like events** — detected from chest optical flow magnitude spikes (visual cue of respiratory illness)
4. **Combined with** PPG-derived breathing rate to generate training labels

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
