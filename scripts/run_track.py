"""CLI: run rocket detect + track on image/video/dir."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts._paths import ROOT, default_source, default_weights, resolve_device
from rocket_track.pipeline import TrackPipeline

PROFILES = {
    "quality": {"imgsz": 640, "conf": 0.25},
    "fast": {"imgsz": 512, "conf": 0.30},
    "realtime": {"imgsz": 416, "conf": 0.35},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detect and track rockets (SORT default).")
    p.add_argument("--source", type=Path, default=default_source(), help="Image, video, or directory")
    p.add_argument("--weights", type=Path, default=None, help="YOLO .pt weights (default: best.pt; nano only for realtime)")
    p.add_argument("--tracker", choices=["sort", "bytetrack"], default="sort")
    p.add_argument("--device", default="auto", help="auto | cpu | 0 | cuda")
    p.add_argument(
        "--profile",
        choices=sorted(PROFILES.keys()),
        default="fast",
        help="quality=640, fast=512 (default), realtime=416",
    )
    p.add_argument("--imgsz", type=int, default=None, help="Override profile imgsz")
    p.add_argument("--conf", type=float, default=None, help="Override profile conf")
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--half", action=argparse.BooleanOptionalAction, default=None,
                   help="FP16 on CUDA (default: on for GPU, off for CPU)")
    p.add_argument("--max-age", type=int, default=30)
    p.add_argument("--min-hits", type=int, default=3)
    p.add_argument("--track-iou", type=float, default=0.3)
    p.add_argument(
        "--coast-frames",
        type=int,
        default=15,
        help="Draw Kalman-predicted boxes for N frames after a miss (0=classic SORT)",
    )
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "tracked")
    p.add_argument("--no-save", action="store_true", help="Skip writing annotated video (speed check)")
    p.add_argument("--warmup", type=int, default=10, help="Frames excluded from infer FPS")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    profile = PROFILES[args.profile]
    imgsz = args.imgsz if args.imgsz is not None else profile["imgsz"]
    conf = args.conf if args.conf is not None else profile["conf"]
    # Nano is only ~1–3 FPS faster than s on a 4060 and loses track hit-rate; use for realtime only.
    prefer_nano = args.profile == "realtime"
    weights = args.weights if args.weights is not None else default_weights(prefer_nano=prefer_nano)

    print(f"source={args.source}")
    print(f"weights={weights}")
    print(f"device={device}  profile={args.profile}  imgsz={imgsz}  conf={conf}  half={args.half}")

    pipe = TrackPipeline(
        weights=weights,
        device=device,
        imgsz=imgsz,
        conf=conf,
        iou=args.iou,
        tracker=args.tracker,
        max_age=args.max_age,
        min_hits=args.min_hits,
        track_iou=args.track_iou,
        coast_frames=args.coast_frames,
        half=args.half,
    )
    stats = pipe.run(
        args.source,
        args.out,
        save_video=not args.no_save,
        warmup=args.warmup,
    )
    hit_rate = (100.0 * stats.frames_with_tracks / stats.n_frames) if stats.n_frames else 0.0
    print(f"Wrote: {stats.out_path}")
    print(
        f"frames={stats.n_frames}  with_tracks={stats.frames_with_tracks} ({hit_rate:.1f}%)  "
        f"elapsed={stats.elapsed_s:.2f}s  e2e_fps={stats.fps:.2f}  "
        f"infer_fps={stats.infer_fps:.2f} (warmup={args.warmup})"
    )


if __name__ == "__main__":
    main()
