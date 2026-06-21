"""
Vision Layer Configuration
==========================
All paths, hyperparameters, and label definitions for the influenza
health monitoring vision pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(
    r"C:\Users\Asus\Desktop\intel multimodal (vision layer)"
)
DATA_ROOT = PROJECT_ROOT / "UBFC_rPPG" / "datasets"
UBFC1_DIR = DATA_ROOT / "UBFC1"
UBFC2_DIR = DATA_ROOT / "UBFC2"

OUTPUT_DIR = PROJECT_ROOT / "vision_layer" / "output"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
ONNX_DIR = OUTPUT_DIR / "onnx"
OPENVINO_DIR = OUTPUT_DIR / "openvino"

# Auto-create output directories
for d in [OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR, PREDICTION_DIR,
          ONNX_DIR, OPENVINO_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Video / windowing
# ---------------------------------------------------------------------------

WINDOW_SEC: float = 10.0         # duration of each sliding window
STRIDE_SEC: float = 5.0          # stride between consecutive windows
TARGET_FPS: int = 30             # frames per second (UBFC native)
FRAME_SIZE: tuple[int, int] = (224, 224)
FRAMES_PER_WINDOW: int = int(WINDOW_SEC * TARGET_FPS)   # 300 frames
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# Detection (YOLO-World or fast fixed-ROI)
# ---------------------------------------------------------------------------

# Set to True to use YOLO-World open-vocabulary detection.
# YOLO-World is accurate but slow (~7s/frame on GPU, ~30s on CPU).
# When False, uses fixed proportional ROIs — near-instant (<1µs) and
# perfectly valid for UBFC since all videos are frontal face recordings.
USE_YOLO_WORLD: bool = False

# ultralytics model identifier
YOLO_MODEL: str = "yolov8s-worldv2"

# Text prompts for YOLO-World open-vocabulary detection
YOLO_CLASSES: list[str] = [
    "face",
    "person face",
    "chest",
    "upper body",
    "torso",
    "mouth open",
    "coughing",
]

# Detection confidence threshold
YOLO_CONF: float = 0.35

# Run detection on every Nth frame of a window (1 = every frame; higher = faster)
YOLO_FRAME_INTERVAL: int = 30   # detect on ~1 frame per second at 30 fps

# ---------------------------------------------------------------------------
# Classification (Swin Transformer)
# ---------------------------------------------------------------------------

SWIN_MODEL: str = "microsoft/swin-tiny-patch4-window7-224"
# Use swin-tiny (28M params) — 3× faster than swin-base (87M).
# Still outputs strong features; better suited for DK-2500 edge deployment.

NUM_CLASSES: int = 3
CLASS_NAMES: dict[int, str] = {
    0: "healthy",
    1: "semi_healthy",
    2: "unhealthy",
}

# Swin-T hidden dim for the pooled feature vector
FEATURE_DIM: int = 768   # 768 for swin-tiny, 1024 for swin-base

# ---------------------------------------------------------------------------
# Label-generation heuristics
# ---------------------------------------------------------------------------

# Breathing-rate (BR) bands in breaths per minute, derived from PPG waveform
BR_HEALTHY_MIN: float = 12.0
BR_HEALTHY_MAX: float = 20.0
BR_TACHYPNEA:    float = 24.0         # rapid breathing threshold
BR_BRADYPNEA:    float = 8.0          # slow breathing threshold

# Cough heuristic: optical-flow magnitude in chest ROI above this percentile
# is considered a "cough-like" event
COUGH_FLOW_PERCENTILE: float = 95.0

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

BATCH_SIZE: int = 4           # small — 300×224×224 frames per sample is heavy
NUM_EPOCHS: int = 50
LEARNING_RATE: float = 1e-4
WEIGHT_DECAY: float = 1e-4
NUM_WORKERS: int = 0   # 0 = main process only (avoids Windows spawn overhead)

# AdamW betas
ADAM_BETA1: float = 0.9
ADAM_BETA2: float = 0.999

# Cosine-annealing scheduler
WARMUP_EPOCHS: int = 3
T_0: int = 10        # first restart after T_0 epochs
T_MULT: int = 2      # each subsequent cycle is T_MULT × longer

# Validation split
VAL_SPLIT: float = 0.2

# Early stopping
EARLY_STOP_PATIENCE: int = 10

# Class imbalance — compute from data or use defaults
CLASS_WEIGHTS: list[float] = [1.0, 1.5, 2.0]  # unhealthy up-weighted

# Focal Loss gamma (0 = plain CrossEntropy)
FOCAL_GAMMA: float = 2.0

# ---------------------------------------------------------------------------
# Deployment
# ---------------------------------------------------------------------------

ONNX_OPSET_VERSION: int = 17
OPENVINO_PRECISION: str = "INT8"     # FP16, FP32 also supported
LATENCY_TARGET_MS: float = 100.0     # DK-2500 inference budget

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

RANDOM_SEED: int = 42
DEVICE: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"
