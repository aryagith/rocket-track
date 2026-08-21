"""Export rocket YOLO weights to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts._paths import ROOT, default_weights


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export rocket detector to ONNX.")
    p.add_argument("--weights", type=Path, default=default_weights())
    p.add_argument("--out", type=Path, default=ROOT / "weights" / "best.onnx")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--dynamic", action="store_true", default=True)
    p.add_argument("--simplify", action="store_true", default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    if not args.weights.exists():
        raise SystemExit(f"Weights not found: {args.weights}")
    model = YOLO(str(args.weights))
    export_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        dynamic=args.dynamic,
        simplify=args.simplify,
    )
    export_path = Path(export_path)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if export_path.resolve() != args.out.resolve():
        args.out.write_bytes(export_path.read_bytes())
        print(f"Copied ONNX to {args.out}")
    print(f"ONNX: {args.out if args.out.exists() else export_path}")


if __name__ == "__main__":
    main()
