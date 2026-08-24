"""Export frames with no confirmed detection for hard-example labeling.

Writes JPEGs under ``outputs/hard_frames/<video_stem>/`` plus a manifest CSV.
Does not invent labels — use these to expand the Roboflow set and retrain.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from scripts._paths import ROOT, default_source, default_weights, resolve_device
from rocket_track.detect import YOLODetector
from rocket_track.pipeline import iter_frames


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dump frames with zero detections (hard negatives/misses).")
    p.add_argument("--source", type=Path, default=default_source())
    p.add_argument("--weights", type=Path, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.20)
    p.add_argument("--max-frames", type=int, default=200, help="Cap exported frames (0=all misses)")
    p.add_argument("--stride", type=int, default=1, help="Only consider every Nth frame")
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "hard_frames")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    weights = args.weights or default_weights(prefer_nano=False)
    det = YOLODetector(weights, device=device, imgsz=args.imgsz, conf=args.conf, half=device != "cpu")
    try:
        det.warmup()
    except Exception:  # noqa: BLE001
        pass

    stem = args.source.stem.replace(" ", "_")
    out_dir = args.out / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.csv"

    rows = []
    exported = 0
    for idx, frame in iter_frames(args.source):
        if args.stride > 1 and (idx % args.stride) != 0:
            continue
        dets = det.predict(frame)
        if dets:
            continue
        fp = out_dir / f"miss_{idx:06d}.jpg"
        cv2.imwrite(str(fp), frame)
        rows.append({"frame": idx, "path": str(fp.relative_to(ROOT))})
        exported += 1
        if args.max_frames and exported >= args.max_frames:
            break

    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "path"])
        w.writeheader()
        w.writerows(rows)
    print(f"Exported {exported} miss frames -> {out_dir}")
    print(f"Manifest: {manifest}")
    print("Label these in Roboflow / Label Studio, merge into train/, then retrain.")


if __name__ == "__main__":
    main()
