"""CLI: run rocket detect + track on image/video/dir."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts._paths import ROOT, default_source, default_weights, resolve_device

# Ensure package import works as `python -m scripts.run_track`
from rocket_track.pipeline import TrackPipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detect and track rockets (SORT default).")
    p.add_argument("--source", type=Path, default=default_source(), help="Image, video, or directory")
    p.add_argument("--weights", type=Path, default=default_weights(), help="YOLO .pt weights")
    p.add_argument("--tracker", choices=["sort", "bytetrack"], default="sort")
    p.add_argument("--device", default="auto", help="auto | cpu | 0 | cuda")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--max-age", type=int, default=30)
    p.add_argument("--min-hits", type=int, default=3)
    p.add_argument("--track-iou", type=float, default=0.3)
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "track")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    pipe = TrackPipeline(
        weights=args.weights,
        device=device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        tracker=args.tracker,
        max_age=args.max_age,
        min_hits=args.min_hits,
        track_iou=args.track_iou,
    )
    out = pipe.run(args.source, args.out)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
