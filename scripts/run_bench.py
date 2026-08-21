"""CLI: benchmark inference backends on rocket media."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts._paths import ROOT, default_onnx, default_source, default_weights
from rocket_track.bench import default_platform_id, run_bench


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark rocket detector backends.")
    p.add_argument("--source", type=Path, default=default_source())
    p.add_argument("--weights", type=Path, default=default_weights())
    p.add_argument(
        "--backends",
        default="pytorch_cpu,pytorch_cuda,onnx_cpu,onnx_cuda,tensorrt",
        help="Comma-separated backend names",
    )
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--max-frames", type=int, default=60)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--platform-id", default=None)
    p.add_argument("--onnx", type=Path, default=default_onnx())
    p.add_argument("--out", type=Path, default=ROOT / "assets" / "results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    platform_id = args.platform_id or default_platform_id()
    result = run_bench(
        source=args.source,
        weights=args.weights,
        backends=backends,
        out_dir=args.out,
        warmup=args.warmup,
        max_frames=args.max_frames,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        platform_id=platform_id,
        onnx_path=args.onnx if args.onnx.exists() else None,
    )
    print(f"Platform: {result['platform_id']}")
    for row in result["rows"]:
        if row.n_frames:
            print(
                f"  {row.backend}: mean={row.mean_ms:.2f} ms  fps={row.fps:.2f}  "
                f"p95={row.p95_ms:.2f}  vram={row.peak_vram_mb}"
            )
        else:
            print(f"  {row.backend}: SKIP — {row.notes}")
    print(f"CSV: {result['csv']}")
    print(f"MD:  {result['md']}")
    print(f"PNG: {result['png']}")


if __name__ == "__main__":
    main()
