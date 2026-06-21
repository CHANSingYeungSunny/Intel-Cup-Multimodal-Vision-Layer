"""
ONNX Export
===========
Export the trained Swin Transformer to ONNX format with dynamic batch
and fixed 224×224 input.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import (
    CHECKPOINT_DIR,
    ONNX_DIR,
    ONNX_OPSET_VERSION,
    DEVICE,
)
from model import build_model


def export_to_onnx(
    checkpoint_path: Path,
    output_path: Path | None = None,
    opset: int = ONNX_OPSET_VERSION,
    dynamic_batch: bool = True,
) -> Path:
    """
    Export the Swin classifier to ONNX.

    Args:
        checkpoint_path  – path to .pt checkpoint
        output_path      – destination .onnx (default: ONNX_DIR / swin_health.onnx)
        opset            – ONNX opset version
        dynamic_batch    – whether to use dynamic batch axis

    Returns the path to the saved .onnx file.
    """
    if output_path is None:
        output_path = ONNX_DIR / "swin_health.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading checkpoint: {checkpoint_path}")
    model = build_model()
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.to("cpu")
    model.eval()

    # Dummy input
    dummy = torch.randn(1, 3, 224, 224, device="cpu")

    # Dynamic axes
    dynamic_axes = {
        "input": {0: "batch"},
        "logits": {0: "batch"},
        "features": {0: "batch"},
    } if dynamic_batch else None

    print(f"Exporting to {output_path} ...")
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits", "features"],
        dynamic_axes=dynamic_axes,
    )

    # Verify
    import onnx
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model verified: {output_path}")

    # Quick shape check
    print(f"  Input  shape: {onnx_model.graph.input[0].type.tensor_type.shape}")
    print(f"  Output shapes:")
    for out in onnx_model.graph.output:
        print(f"    {out.name}: {out.type.tensor_type.shape}")

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Swin Transformer to ONNX")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .pt checkpoint (default: best_model.pt)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to output .onnx")
    parser.add_argument("--opset", type=int, default=ONNX_OPSET_VERSION)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint) if args.checkpoint else CHECKPOINT_DIR / "best_model.pt"
    if not ckpt.exists():
        ckpt = CHECKPOINT_DIR / "last_model.pt"

    out = Path(args.output) if args.output else None
    export_to_onnx(ckpt, out, opset=args.opset)
