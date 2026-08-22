"""Compare detect+track infer FPS across weights/profiles."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from scripts._paths import ROOT, default_source, resolve_device
from rocket_track.pipeline import TrackPipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare tracking infer FPS across weight files.")
    p.add_argument("--source", type=Path, default=default_source())
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--weights",
        nargs="+",
        default=[],
        help="Weight paths (default: weights/best.pt and weights/best_n.pt if present)",
    )
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--out", type=Path, default=ROOT / "assets" / "results" / "speed_compare.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    weights = [Path(w) for w in args.weights]
    if not weights:
        for cand in [ROOT / "weights" / "best.pt", ROOT / "weights" / "best_n.pt"]:
            if cand.exists():
                weights.append(cand)
    if not weights:
        raise SystemExit("No weights found")

    rows = []
    for w in weights:
        print(f"=== {w} device={device} imgsz={args.imgsz} ===")
        pipe = TrackPipeline(
            weights=w,
            device=device,
            imgsz=args.imgsz,
            conf=args.conf,
            half=device != "cpu",
            tracker="sort",
            min_hits=1,
        )
        stats = pipe.run(
            args.source,
            ROOT / "outputs" / "speed_compare",
            save_video=False,
            warmup=args.warmup,
        )
        row = {
            "weights": w.name,
            "device": device,
            "imgsz": args.imgsz,
            "n_frames": stats.n_frames,
            "with_tracks": stats.frames_with_tracks,
            "e2e_fps": round(stats.fps, 2),
            "infer_fps": round(stats.infer_fps, 2),
        }
        rows.append(row)
        print(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
