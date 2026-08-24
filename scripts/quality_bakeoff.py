"""Bake-off detect+track quality knobs on a video; write CSV of hit-rates."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from scripts._paths import ROOT, default_source, default_weights, resolve_device
from rocket_track.pipeline import TrackPipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quality bake-off: profile/conf/coast hit-rate.")
    p.add_argument("--source", type=Path, default=default_source())
    p.add_argument("--weights", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=ROOT / "assets" / "results" / "quality_bakeoff.csv")
    return p.parse_args()


# (name, imgsz, conf, coast_frames, min_hits)
COMBOS = [
    ("fast_c30_coast0", 512, 0.30, 0, 3),
    ("fast_c30_coast15", 512, 0.30, 15, 3),
    ("fast_c25_coast15", 512, 0.25, 15, 3),
    ("fast_c20_coast15", 512, 0.20, 15, 3),
    ("fast_c25_coast25", 512, 0.25, 25, 3),
    ("quality_c25_coast15", 640, 0.25, 15, 3),
    ("quality_c20_coast15", 640, 0.20, 15, 3),
    ("quality_c20_coast25", 640, 0.20, 25, 3),
    ("quality_c15_coast25", 640, 0.15, 25, 3),
]


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    weights = args.weights or default_weights(prefer_nano=False)
    rows = []
    for name, imgsz, conf, coast, min_hits in COMBOS:
        print(f"=== {name} imgsz={imgsz} conf={conf} coast={coast} ===")
        pipe = TrackPipeline(
            weights=weights,
            device=device,
            imgsz=imgsz,
            conf=conf,
            half=device != "cpu",
            tracker="sort",
            min_hits=min_hits,
            coast_frames=coast,
        )
        stats = pipe.run(args.source, ROOT / "outputs" / "quality_bakeoff", save_video=False, warmup=10)
        hit = (100.0 * stats.frames_with_tracks / stats.n_frames) if stats.n_frames else 0.0
        row = {
            "name": name,
            "weights": Path(weights).name,
            "imgsz": imgsz,
            "conf": conf,
            "coast_frames": coast,
            "min_hits": min_hits,
            "n_frames": stats.n_frames,
            "with_tracks": stats.frames_with_tracks,
            "hit_pct": round(hit, 1),
            "e2e_fps": round(stats.fps, 2),
            "infer_fps": round(stats.infer_fps, 2),
        }
        rows.append(row)
        print(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    best = max(rows, key=lambda r: (r["hit_pct"], r["infer_fps"]))
    print(f"Wrote {args.out}")
    print(f"Best by hit_pct: {best['name']} -> {best['hit_pct']}% @ infer {best['infer_fps']} FPS")


if __name__ == "__main__":
    main()
