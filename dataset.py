"""
Video Slice Dataset
===================
Iterates over UBFC1/UBFC2 subject folders, decodes vid.avi files, and
extracts 10-second sliding windows with 5-second stride.  Frames are
resized to 224×224 and normalised with ImageNet statistics.

Returns a dict:
    frames  : (T, 3, H, W) float32 tensor
    label   : int  {0,1,2}
    window_id : str  e.g. "UBFC1/5-gt/win_000"
    subject  : str  e.g. "UBFC1/5-gt"
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import (
    UBFC1_DIR,
    UBFC2_DIR,
    WINDOW_SEC,
    STRIDE_SEC,
    TARGET_FPS,
    FRAME_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    FRAMES_PER_WINDOW,
    NUM_WORKERS,
    RANDOM_SEED,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _list_subjects() -> list[dict]:
    """Return all (subject_path, source_dataset, ground_truth_path, vid_path)."""
    subjects: list[dict] = []

    for src_dir, src_name in [(UBFC1_DIR, "UBFC1"), (UBFC2_DIR, "UBFC2")]:
        if not src_dir.exists():
            continue
        for sub_dir in sorted(src_dir.iterdir()):
            if not sub_dir.is_dir():
                continue
            vid = sub_dir / "vid.avi"
            if not vid.exists():
                continue

            # UBFC1 uses gtdump.xmp; UBFC2 uses ground_truth.txt
            gt = sub_dir / "gtdump.xmp"
            if not gt.exists():
                gt = sub_dir / "ground_truth.txt"
            if not gt.exists():
                gt = None

            subjects.append({
                "name": f"{src_name}/{sub_dir.name}",
                "vid": vid,
                "gt": gt,
                "source": src_name,
            })

    return subjects


def _load_ground_truth(gt_path: Path, num_frames: int) -> np.ndarray:
    """
    Load the PPG ground-truth signal, resampled to exactly *num_frames*.
    UBFC1 gtdump.xmp:  CSV  → column 1 = PPG
    UBFC2 ground_truth.txt:  space-delimited floats → first column ≈ PPG
    """
    if gt_path is None or not gt_path.exists():
        return np.zeros(num_frames, dtype=np.float32)

    text = gt_path.read_text(encoding="utf-8").strip()
    lines = text.splitlines()
    values: list[float] = []

    if gt_path.suffix == ".xmp" or gt_path.name == "gtdump.xmp":
        # CSV: timestamp, hr, spo2, ppg_value, ...  (column 3 or 4)
        for line in lines:
            parts = line.split(",")
            if len(parts) >= 4:
                try:
                    values.append(float(parts[3]))   # PPG column
                except ValueError:
                    continue
    else:
        # Space-delimited: each line is one or more float values
        for line in lines:
            for token in line.split():
                try:
                    values.append(float(token))
                except ValueError:
                    continue

    if not values:
        return np.zeros(num_frames, dtype=np.float32)

    signal = np.array(values, dtype=np.float32)

    # Resample to target length
    if len(signal) == num_frames:
        return signal
    indices = np.linspace(0, len(signal) - 1, num_frames)
    return np.interp(indices, np.arange(len(signal)), signal).astype(np.float32)


# ---------------------------------------------------------------------------
# Main Dataset
# ---------------------------------------------------------------------------

class UBFCWindowDataset(Dataset):
    """
    PyTorch Dataset that yields sliding windows from UBFC videos.

    Each item:
        {
            "frames":    torch.Tensor (T, 3, H, W),
            "ppg":       torch.Tensor (T,)    — ground-truth PPG for the window,
            "window_id": str,
            "subject":   str,
            "label":     int  (filled by preprocessing or set to -1 initially),
        }
    """

    def __init__(
        self,
        subjects: list[dict] | None = None,
        cache_dir: Path | None = None,
    ):
        super().__init__()
        self.subjects = subjects if subjects is not None else _list_subjects()
        self.cache_dir = cache_dir or (Path(__file__).parent / "output" / "window_cache")

        # Pre-compute window index
        self._windows: list[dict] = []
        self._build_index()

    # ------------------------------------------------------------------
    def _build_index(self) -> None:
        """Scan every subject video and record (subject_idx, start_frame)."""
        print(f"Building window index from {len(self.subjects)} subjects ...")
        t0 = time.time()

        for sub_idx, sub in enumerate(self.subjects):
            cap = cv2.VideoCapture(str(sub["vid"]))
            if not cap.isOpened():
                print(f"  WARNING: cannot open {sub['vid']}")
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            if fps <= 0:
                fps = TARGET_FPS

            window_len = int(WINDOW_SEC * fps)
            stride_len = int(STRIDE_SEC * fps)

            if total_frames < window_len:
                continue

            start = 0
            while start + window_len <= total_frames:
                self._windows.append({
                    "sub_idx": sub_idx,
                    "start_frame": start,
                    "window_len": window_len,
                    "fps": fps,
                })
                start += stride_len

        elapsed = time.time() - t0
        print(f"  -> {len(self._windows)} windows ({elapsed:.1f} s)")

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._windows)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        win = self._windows[idx]
        sub = self.subjects[win["sub_idx"]]
        window_id = f"{sub['name']}/win_{win['start_frame']:06d}"

        # Load from JPEG cache if available (near-instant), else read from video
        cache_path = (Path(__file__).parent / "output" / "frame_cache" /
                      f"{window_id.replace('/', '_')}.jpg")

        if cache_path.exists():
            frame = cv2.imread(str(cache_path))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            mid_frame_num = win["start_frame"] + win["window_len"] // 2
            cap = cv2.VideoCapture(str(sub["vid"]))
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_num)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                frame = np.zeros((*FRAME_SIZE, 3), dtype=np.uint8)
            else:
                frame = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_LINEAR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to tensor: (1, C, H, W)
        frame_arr = frame.astype(np.float32) / 255.0
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std = np.array(IMAGENET_STD, dtype=np.float32)
        frame_arr = (frame_arr - mean) / std
        frame_tensor = torch.from_numpy(frame_arr).permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W

        # PPG — still load full signal for label generation (used in preprocessing only)
        ppg = _load_ground_truth(sub["gt"], win["window_len"])
        if len(ppg) > FRAMES_PER_WINDOW:
            ppg_indices = np.linspace(0, len(ppg) - 1, FRAMES_PER_WINDOW, dtype=int)
            ppg = ppg[ppg_indices]
        elif len(ppg) < FRAMES_PER_WINDOW:
            ppg = np.pad(ppg, (0, FRAMES_PER_WINDOW - len(ppg)), mode="edge")

        window_id = f"{sub['name']}/win_{win['start_frame']:06d}"

        return {
            "frames": frame_tensor,            # (1, 3, 224, 224)
            "ppg": torch.from_numpy(ppg),      # (T,)
            "window_id": window_id,
            "subject": sub["name"],
            "label": -1,                        # to be filled by preprocessing
            "sub_idx": win["sub_idx"],
            "start_frame": win["start_frame"],
            "window_len": win["window_len"],
            "vid_path": str(sub["vid"]),
        }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_dataloaders(
    batch_size: int = 4,
    val_split: float = 0.2,
    num_workers: int = NUM_WORKERS,
    seed: int = RANDOM_SEED,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train + val DataLoaders with a deterministic subject-level split.
    """
    subjects = _list_subjects()
    n = len(subjects)
    indices = list(range(n))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    split = int(n * (1 - val_split))
    train_subjects = [subjects[i] for i in indices[:split]]
    val_subjects = [subjects[i] for i in indices[split:]]

    print(f"Train subjects: {len(train_subjects)}  |  Val subjects: {len(val_subjects)}")

    train_ds = UBFCWindowDataset(train_subjects)
    val_ds = UBFCWindowDataset(val_subjects)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_collate_windows,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_collate_windows,
    )

    return train_loader, val_loader


def _collate_windows(batch: list[dict]) -> dict:
    """Custom collate — stacks frames and PPG, keeps metadata as lists."""
    frames = torch.stack([item["frames"] for item in batch], dim=0)   # B,T,C,H,W
    ppg = torch.stack([item["ppg"] for item in batch], dim=0)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    window_ids = [item["window_id"] for item in batch]
    subjects = [item["subject"] for item in batch]
    return {
        "frames": frames,
        "ppg": ppg,
        "label": labels,
        "window_id": window_ids,
        "subject": subjects,
    }


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ds = UBFCWindowDataset()
    print(f"Total windows: {len(ds)}")

    if len(ds) > 0:
        sample = ds[0]
        print(f"  frames:  {sample['frames'].shape}   dtype={sample['frames'].dtype}")
        print(f"  ppg:     {sample['ppg'].shape}")
        print(f"  window:  {sample['window_id']}")
        print(f"  subject: {sample['subject']}")
    else:
        print("  No windows found — check dataset paths.")
