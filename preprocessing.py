"""
Preprocessing & Label Generation
================================
Fast face/chest ROI detection (fixed crops — UBFC is all frontal video),
PPG-based breathing frequency, optical-flow cough detection, and heuristic
label generation (healthy / semi_healthy / unhealthy).
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt, hilbert
from tqdm import tqdm

from config import (
    FRAME_SIZE,
    TARGET_FPS,
    BR_HEALTHY_MIN,
    BR_HEALTHY_MAX,
    BR_TACHYPNEA,
    BR_BRADYPNEA,
    COUGH_FLOW_PERCENTILE,
    CLASS_NAMES,
    OUTPUT_DIR,
)


# ---------------------------------------------------------------------------
# Fast ROI detection (fixed crops — UBFC is 100% frontal face video)
# ---------------------------------------------------------------------------

_ROI_CACHE: dict[str, dict] = {}   # per-video cached ROIs


# Lazy-loaded YOLO-World model (only if USE_YOLO_WORLD = True)
_yolo_model = None

def _get_yolo():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        from config import YOLO_MODEL, YOLO_CLASSES
        _yolo_model = YOLO(YOLO_MODEL)
        _yolo_model.set_classes(YOLO_CLASSES)
    return _yolo_model


def detect_regions(frame: np.ndarray, vid_path: str = "") -> dict:
    """
    Detect face and chest regions.

    If USE_YOLO_WORLD=True: uses YOLO-World open-vocabulary detector.
    If USE_YOLO_WORLD=False: uses fixed proportional ROIs (<1µs).
      UBFC videos are frontal-face — fixed crops are faster *and* more reliable.

    Returns:  {"face": (x1,y1,x2,y2), "chest": (x1,y1,x2,y2)}
    """
    from config import USE_YOLO_WORLD, YOLO_CONF

    h, w = frame.shape[:2]

    if USE_YOLO_WORLD:
        try:
            model = _get_yolo()
            results = model(frame, conf=YOLO_CONF, verbose=False)
            face_box = None
            chest_box = None
            if results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                names = results[0].names
                for box, cid in zip(boxes, cls_ids):
                    name = names.get(cid, "").lower()
                    if any(k in name for k in ("face", "person face")) and face_box is None:
                        face_box = tuple(box.astype(int).tolist())
                    elif any(k in name for k in ("chest", "upper body", "torso")) and chest_box is None:
                        chest_box = tuple(box.astype(int).tolist())
            if face_box and chest_box:
                return {"face": face_box, "chest": chest_box}
            # Fall through to fixed ROIs if YOLO misses
        except Exception:
            pass  # Fall through to fixed ROIs

    # Fixed proportional ROIs (default)
    face = (
        int(w * 0.28), int(h * 0.05),
        int(w * 0.72), int(h * 0.60),
    )
    chest = (
        int(w * 0.25), int(h * 0.55),
        int(w * 0.75), int(h * 0.90),
    )
    return {"face": face, "chest": chest}


def crop_region(frame: np.ndarray, bbox, target_size=FRAME_SIZE) -> np.ndarray:
    """Crop frame to bbox (x1,y1,x2,y2) and resize."""
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
    crop = frame[y1:y2, x1:x2]
    return cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# PPG signal processing — breathing frequency estimation
# ---------------------------------------------------------------------------

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return b, a


def estimate_breathing_rate(ppg_signal: np.ndarray, fs: float = TARGET_FPS) -> float:
    """Estimate breathing rate (breaths/min) from PPG waveform envelope."""
    if ppg_signal.std() < 1e-6:
        return -1.0
    ppg = ppg_signal - np.mean(ppg_signal)
    try:
        b, a = butter_bandpass(0.1, 0.5, fs, order=4)
        resp = filtfilt(b, a, ppg)
    except Exception:
        return -1.0
    analytic = hilbert(resp)
    envelope = np.abs(analytic)
    height = np.percentile(envelope, 50)
    distance = int(fs * 1.0)
    peaks, _ = find_peaks(envelope, height=height, distance=distance)
    if len(peaks) < 2:
        return -1.0
    ibi = np.diff(peaks) / fs
    br_instantaneous = 60.0 / ibi
    lo, hi = np.percentile(br_instantaneous, [10, 90])
    br_filtered = br_instantaneous[(br_instantaneous >= lo) & (br_instantaneous <= hi)]
    if len(br_filtered) == 0:
        return -1.0
    return float(np.median(br_filtered))


# ---------------------------------------------------------------------------
# Cough detection via optical flow in chest ROI
# ---------------------------------------------------------------------------

def detect_cough_events(frames: np.ndarray, chest_bbox) -> int:
    """
    Count cough-like events from optical flow magnitude in chest ROI.
    frames: (T, H, W, 3) uint8 RGB array.
    """
    if len(frames) < 2:
        return 0

    x1, y1, x2, y2 = chest_bbox
    prev_gray = cv2.cvtColor(frames[0][y1:y2, x1:x2], cv2.COLOR_RGB2GRAY)
    flow_magnitudes = []

    for t in range(1, len(frames)):
        gray = cv2.cvtColor(frames[t][y1:y2, x1:x2], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag = np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2))
        flow_magnitudes.append(mag)
        prev_gray = gray

    if not flow_magnitudes:
        return 0

    flow_mag = np.array(flow_magnitudes)
    threshold = np.percentile(flow_mag, COUGH_FLOW_PERCENTILE)
    cough_frames = flow_mag > threshold

    events = 0
    run = 0
    for cough in cough_frames:
        if cough:
            run += 1
        else:
            if run >= 3:
                events += 1
            run = 0
    if run >= 3:
        events += 1

    return events


# ---------------------------------------------------------------------------
# Label generation (breathing rate + cough events)
# ---------------------------------------------------------------------------

def generate_label(breathing_rate: float, cough_events: int) -> tuple[int, dict]:
    """
    Heuristic: combine breathing rate and cough events.

    healthy       : 12 <= BR <= 20,  no cough
    semi_healthy  : borderline BR or 1-2 coughs
    unhealthy     : BR < 8 or > 24,  or >= 3 coughs
    """
    br = breathing_rate
    cough = cough_events
    score = 0
    reasons = []

    if br < 0:
        reasons.append("br_unavailable")
        score += 1
    elif br < BR_BRADYPNEA:
        reasons.append(f"bradypnea_br={br:.1f}")
        score += 2
    elif br < BR_HEALTHY_MIN:
        reasons.append(f"borderline_low_br={br:.1f}")
        score += 1
    elif br <= BR_HEALTHY_MAX:
        reasons.append(f"normal_br={br:.1f}")
    elif br <= BR_TACHYPNEA:
        reasons.append(f"borderline_high_br={br:.1f}")
        score += 1
    else:
        reasons.append(f"tachypnea_br={br:.1f}")
        score += 2

    if cough >= 3:
        reasons.append(f"frequent_cough={cough}")
        score += 2
    elif cough >= 1:
        reasons.append(f"occasional_cough={cough}")
        score += 1

    if score >= 2:
        label = 2
    elif score == 1:
        label = 1
    else:
        label = 0

    debug = {
        "breathing_rate": round(br, 2),
        "cough_events": cough,
        "score": score,
        "reasons": "; ".join(reasons),
        "label_name": CLASS_NAMES[label],
    }
    return label, debug


# ---------------------------------------------------------------------------
# Batch preprocessing
# ---------------------------------------------------------------------------

def preprocess_batch(frames, ppg_signals, window_ids):
    """
    Full preprocessing for a batch of windows.

    frames:  (B, T, H, W, 3) uint8
    ppg_signals: (B, T)
    Returns: swin_input (B, 3, 224, 224), labels (B,), metadata list
    """
    B = frames.shape[0]
    swin_frames = np.zeros((B, 3, *FRAME_SIZE), dtype=np.float32)
    labels = np.zeros(B, dtype=np.int64)
    meta_list = []

    mid = frames.shape[1] // 2

    for b in range(B):
        mid_frame = frames[b, mid]
        regions = detect_regions(mid_frame)
        face_crop = crop_region(mid_frame, regions["face"])
        face_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
        face_f32 = face_bgr.astype(np.float32) / 255.0
        face_f32 = (face_f32 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        swin_frames[b] = face_f32.transpose(2, 0, 1)

        br = estimate_breathing_rate(ppg_signals[b])
        cough = detect_cough_events(frames[b], regions["chest"])
        label, debug = generate_label(br, cough)
        labels[b] = label
        debug["window_id"] = window_ids[b]
        meta_list.append(debug)

    return swin_frames, labels, meta_list


# ---------------------------------------------------------------------------
# Full dataset label generation
# ---------------------------------------------------------------------------

def preprocess_and_cache_labels(dataset, cache_path=None):
    """Generate labels for all windows. Writes labels.csv."""
    if cache_path is None:
        cache_path = OUTPUT_DIR / "labels.csv"

    rows = []
    print(f"Generating labels for {len(dataset)} windows (PPG + optical flow cough)...")

    for i in tqdm(range(len(dataset)), desc="Label gen"):
        sample = dataset[i]

        # Read video frames for cough detection
        vid_path = sample["vid_path"]
        start = sample["start_frame"]
        win_len = min(sample["window_len"], 90)

        cap = cv2.VideoCapture(vid_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        raw_frames = []
        for _ in range(win_len):
            ret, f = cap.read()
            if not ret:
                break
            raw_frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()

        ppg = sample["ppg"].numpy()
        br = estimate_breathing_rate(ppg)

        cough = 0
        if len(raw_frames) >= 2:
            frames_arr = np.stack(raw_frames)
            regions = detect_regions(frames_arr[0], vid_path)
            try:
                cough = detect_cough_events(frames_arr, regions["chest"])
            except Exception:
                pass

        label, debug = generate_label(br, cough)

        rows.append({
            "window_id": sample["window_id"],
            "subject": sample["subject"],
            "label": label,
            "label_name": debug["label_name"],
            "breathing_rate": debug["breathing_rate"],
            "cough_events": debug["cough_events"],
            "face_detected": True,   # fixed ROI always captures face
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(cache_path, index=False)
    counts = df["label_name"].value_counts()
    print(f"Label distribution:\n{counts}")
    print(f"Saved labels to {cache_path}")
    return cache_path


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from dataset import UBFCWindowDataset
    ds = UBFCWindowDataset()
    preprocess_and_cache_labels(ds)
