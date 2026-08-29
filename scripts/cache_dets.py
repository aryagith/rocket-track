"""Run the detector once over a clip and cache its output for replay.

Everything downstream — filter tuning, evaluation, tests — reads the cache
instead of the network, so iteration needs no GPU and finishes in milliseconds.

    python -m scripts.cache_dets --source testing_media/rocket_launch.mov --device cuda
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from scripts._paths import ROOT, default_source, default_weights, resolve_device
from rocket_track.detcache import CacheMeta, FrameDetections, write_cache
from rocket_track.detect import YOLODetector, detections_to_array
from rocket_track.pipeline import is_video, iter_frames, probe_video_fps

CACHE_DIR = ROOT / "assets" / "caches"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cache detector output for a clip.")
    p.add_argument("--source", type=Path, default=default_source())
    p.add_argument("--weights", type=Path, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--conf", type=float, default=0.30)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--max-area-frac", type=float, default=0.04)
    p.add_argument("--min-area-frac", type=float, default=0.0)
    p.add_argument("--max-det", type=int, default=2)
    p.add_argument("--fps", type=float, default=None, help="Override timestamps source FPS")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def default_out(source: Path, weights: Path, imgsz: int, conf: float) -> Path:
    stem = source.stem.replace(" ", "_")
    return CACHE_DIR / f"{stem}_{weights.stem}_{imgsz}_c{int(round(conf * 100)):02d}.jsonl"


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    weights = args.weights or default_weights(prefer_nano=False)
    out = args.out or default_out(args.source, Path(weights), args.imgsz, args.conf)

    fps = args.fps or (probe_video_fps(args.source) if is_video(args.source) else 30.0)
    print(f"source={args.source}  weights={weights}  device={device}")
    print(f"imgsz={args.imgsz}  conf={args.conf}  max_area_frac={args.max_area_frac}  fps={fps}")

    det = YOLODetector(
        weights,
        device=device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        half=device != "cpu",
        max_area_frac=args.max_area_frac,
        min_area_frac=args.min_area_frac,
        max_det=args.max_det,
    )
    try:
        det.warmup()
    except Exception:  # noqa: BLE001
        pass

    records = []
    n_dets = 0
    frame_w = frame_h = 0
    t0 = time.perf_counter()
    for idx, frame in iter_frames(args.source):
        if not frame_w:
            frame_h, frame_w = frame.shape[:2]
        dets = detections_to_array(det.predict(frame))
        n_dets += int(dets.shape[0])
        records.append(FrameDetections(index=idx, t=idx / fps, dets=dets))
    elapsed = time.perf_counter() - t0

    meta = CacheMeta(
        source=str(args.source).replace("\\", "/"),
        weights=str(weights).replace("\\", "/"),
        imgsz=int(args.imgsz),
        conf=float(args.conf),
        fps=float(fps),
        n_frames=len(records),
        frame_w=int(frame_w),
        frame_h=int(frame_h),
    )
    write_cache(out, meta, records)

    frames_with = sum(1 for r in records if r.dets.shape[0] > 0)
    print(
        f"Wrote {out}\n"
        f"frames={len(records)}  frames_with_dets={frames_with}  total_dets={n_dets}  "
        f"detect_elapsed={elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
