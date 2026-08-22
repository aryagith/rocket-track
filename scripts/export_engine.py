"""Export rocket YOLO weights to a TensorRT engine (GPU-local).

Requires TensorRT 10.x matching your CUDA (e.g. ``pip install tensorrt==10.3.0``).
Engines are GPU-specific — rebuild on each machine; do not commit ``*.engine``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts._paths import ROOT, default_weights


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export rocket detector to TensorRT engine.")
    p.add_argument("--weights", type=Path, default=default_weights())
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default="0")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    if not args.weights.exists():
        raise SystemExit(f"Weights not found: {args.weights}")
    try:
        import tensorrt as trt  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "TensorRT Python package not found. On this CUDA 12 stack use:\n"
            "  pip install tensorrt==10.3.0\n"
            "(TensorRT 11+ breaks Ultralytics EXPLICIT_BATCH export as of 8.3.x)"
        ) from e

    model = YOLO(str(args.weights))
    out = model.export(
        format="engine",
        imgsz=args.imgsz,
        half=args.half,
        device=args.device,
    )
    print(f"Engine: {out}")
    print(
        "Speed check:\n"
        f"  python -m scripts.run_track --weights {out} --device cuda "
        f"--imgsz {args.imgsz} --half --no-save"
    )


if __name__ == "__main__":
    main()
