"""
OpenVINO Optimisation
=====================
Convert the ONNX model to OpenVINO IR, apply INT8 quantization, and
benchmark inference latency on CPU (DK-2500 target).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import (
    ONNX_DIR,
    OPENVINO_DIR,
    CHECKPOINT_DIR,
    OPENVINO_PRECISION,
    LATENCY_TARGET_MS,
    BATCH_SIZE,
    NUM_WORKERS,
    FEATURE_DIM,
    NUM_CLASSES,
    DEVICE,
)
from dataset import create_dataloaders, _collate_windows, UBFCWindowDataset
from export_onnx import export_to_onnx


# ---------------------------------------------------------------------------
# OpenVINO conversion
# ---------------------------------------------------------------------------

def convert_to_openvino(
    onnx_path: Path,
    output_dir: Path | None = None,
    precision: str = OPENVINO_PRECISION,
) -> tuple[Path, Path]:
    """
    Convert ONNX → OpenVINO IR (.xml + .bin).

    Returns (xml_path, bin_path).
    """
    if output_dir is None:
        output_dir = OPENVINO_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    xml_path = output_dir / "swin_health.xml"
    bin_path = output_dir / "swin_health.bin"

    print(f"Converting ONNX → OpenVINO IR ({precision}) ...")
    import openvino as ov
    core = ov.Core()

    # Read ONNX
    model = core.read_model(str(onnx_path))

    # Quantize if requested
    if precision.upper() == "INT8":
        print("  Applying INT8 quantization ...")
        # Pot (post-training quantization) with a calibration dataset
        calib_data = _build_calibration_dataset()

        from openvino.tools import pot  # type: ignore
        # Note: full POT calibration requires a config JSON.
        # For now we use NNCF-based PTQ, which is simpler:
        import nncf
        calib_loader = _calib_loader(calib_data)
        quantized = nncf.quantize(model, calib_loader, preset="mixed")
        model = quantized

    elif precision.upper() == "FP16":
        from openvino.runtime import serialize
        # FP16 is the default for many OV conversions; keep as-is.

    # Serialize
    ov.serialize(model, str(xml_path), str(bin_path))

    print(f"  IR saved:  {xml_path}")
    print(f"  Weights:   {bin_path}")

    return xml_path, bin_path


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def _build_calibration_dataset(n_samples: int = 200) -> np.ndarray:
    """Build a small numpy array of calibration frames from the dataset."""
    ds = UBFCWindowDataset()
    frames_list: list[np.ndarray] = []

    indices = np.linspace(0, len(ds) - 1, min(n_samples, len(ds)), dtype=int)
    for i in indices:
        sample = ds[i]
        frames_t = sample["frames"]           # (T, C, H, W)
        mid = frames_t.size(0) // 2
        frame = frames_t[mid].numpy()          # (C, H, W)
        frames_list.append(frame)

    return np.stack(frames_list, axis=0)       # (N, 3, 224, 224)


def _calib_loader(data: np.ndarray) -> "nncf.Dataset":
    """Wrap calibration data for NNCF."""
    import nncf
    def transform_fn(data_item):
        # NNCF expects (1, C, H, W) tensors
        return torch.from_numpy(data_item).unsqueeze(0)
    return nncf.Dataset(data, transform_fn)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark_openvino(
    xml_path: Path,
    n_warmup: int = 20,
    n_iter: int = 100,
    target_ms: float = LATENCY_TARGET_MS,
) -> dict:
    """
    Benchmark OpenVINO inference latency.

    Returns a dict with mean / p50 / p95 / p99 latencies.
    """
    print(f"\nBenchmarking {xml_path.name} ...")

    import openvino as ov
    core = ov.Core()
    compiled = core.compile_model(str(xml_path), "CPU")
    infer = compiled.create_infer_request()

    # Dummy input
    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)

    # Warmup
    for _ in range(n_warmup):
        infer.infer({"input": dummy})

    # Measure
    latencies: list[float] = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        infer.infer({"input": dummy})
        elapsed = (time.perf_counter() - t0) * 1000   # ms
        latencies.append(elapsed)

    lat_arr = np.array(latencies)
    results = {
        "mean_ms": float(lat_arr.mean()),
        "std_ms": float(lat_arr.std()),
        "p50_ms": float(np.percentile(lat_arr, 50)),
        "p95_ms": float(np.percentile(lat_arr, 95)),
        "p99_ms": float(np.percentile(lat_arr, 99)),
        "min_ms": float(lat_arr.min()),
        "max_ms": float(lat_arr.max()),
        "target_ms": target_ms,
        "pass": bool(lat_arr.mean() < target_ms),
    }

    print(f"  Mean  : {results['mean_ms']:.2f} ms")
    print(f"  P50   : {results['p50_ms']:.2f} ms")
    print(f"  P95   : {results['p95_ms']:.2f} ms")
    print(f"  P99   : {results['p99_ms']:.2f} ms")
    print(f"  Target: {target_ms} ms  →  {'PASS' if results['pass'] else 'FAIL'}")

    # Also benchmark ONNX Runtime for comparison
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(
            str(ONNX_DIR / "swin_health.onnx"),
            providers=["CPUExecutionProvider"],
        )
        lat_ort: list[float] = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            sess.run(None, {"input": dummy})
            lat_ort.append((time.perf_counter() - t0) * 1000)
        print(f"\n  ONNX Runtime mean: {np.mean(lat_ort):.2f} ms")
    except Exception:
        pass

    return results


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

def optimize_pipeline(
    checkpoint_path: Path | None = None,
    precision: str = OPENVINO_PRECISION,
) -> dict:
    """
    Full pipeline: ONNX export → OpenVINO IR → INT8 quant → benchmark.
    """
    # Step 1 — ONNX
    ckpt = checkpoint_path or (CHECKPOINT_DIR / "best_model.pt")
    if not ckpt.exists():
        ckpt = CHECKPOINT_DIR / "last_model.pt"

    onnx_path = ONNX_DIR / "swin_health.onnx"
    if not onnx_path.exists():
        onnx_path = export_to_onnx(ckpt, onnx_path)

    # Step 2 — OpenVINO IR
    xml_path, bin_path = convert_to_openvino(onnx_path, precision=precision)

    # Step 3 — Benchmark
    results = benchmark_openvino(xml_path)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenVINO optimisation pipeline")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--onnx", type=str, default=None,
                        help="Path to existing .onnx (skip export step)")
    parser.add_argument("--precision", type=str, default=OPENVINO_PRECISION,
                        choices=["FP32", "FP16", "INT8"])
    parser.add_argument("--benchmark-only", action="store_true",
                        help="Only benchmark an existing IR")
    args = parser.parse_args()

    if args.benchmark_only:
        xml = OPENVINO_DIR / "swin_health.xml"
        benchmark_openvino(xml)
    elif args.onnx:
        xml_path, _ = convert_to_openvino(Path(args.onnx), precision=args.precision)
        benchmark_openvino(xml_path)
    else:
        optimize_pipeline(
            checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
            precision=args.precision,
        )
