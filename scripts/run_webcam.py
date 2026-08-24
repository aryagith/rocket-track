"""Live webcam rocket track — point the camera at a phone playing launch footage.

Keys:
  q / ESC  quit
  r        reset tracker IDs
  s        save a still under outputs/webcam/
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from scripts._paths import ROOT, default_weights, resolve_device
from rocket_track.pipeline import TrackPipeline
from rocket_track.viz import draw_tracks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live webcam detect+SORT for rocket testing.")
    p.add_argument("--camera", type=int, default=0, help="OpenCV camera index (0=default webcam)")
    p.add_argument("--weights", type=Path, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--conf", type=float, default=0.30)
    p.add_argument("--coast-frames", type=int, default=15)
    p.add_argument("--max-area-frac", type=float, default=0.04)
    p.add_argument("--max-det", type=int, default=2)
    p.add_argument("--width", type=int, default=1280, help="Request capture width (0=leave default)")
    p.add_argument("--height", type=int, default=720, help="Request capture height (0=leave default)")
    p.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=True,
                   help="Mirror preview horizontally (default on)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    weights = args.weights or default_weights(prefer_nano=False)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)  # CAP_DSHOW is more reliable on Windows
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")

    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    pipe = TrackPipeline(
        weights=weights,
        device=device,
        imgsz=args.imgsz,
        conf=args.conf,
        half=device != "cpu",
        tracker="sort",
        coast_frames=args.coast_frames,
        max_area_frac=args.max_area_frac,
        max_det=args.max_det,
    )

    print(f"webcam={args.camera}  weights={weights}  device={device}  imgsz={args.imgsz}  conf={args.conf}")
    print("Hold a phone with rocket footage in view.  q=quit  r=reset  s=save still")

    win = "rocket-track webcam"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    still_dir = ROOT / "outputs" / "webcam"
    still_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    t0 = time.perf_counter()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Camera read failed; exiting.")
            break
        if args.mirror:
            frame = cv2.flip(frame, 1)

        tracks = pipe.process_frame(frame)
        vis = draw_tracks(frame, tracks)
        n += 1
        elapsed = time.perf_counter() - t0
        if elapsed >= 0.5:
            fps = n / elapsed
            n = 0
            t0 = time.perf_counter()

        hud = f"FPS {fps:.1f}  tracks {len(tracks)}  [q]uit [r]eset [s]ave"
        cv2.putText(vis, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(vis, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.imshow(win, vis)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            pipe.reset()
            print("Tracker reset")
        if key == ord("s"):
            path = still_dir / f"still_{int(time.time())}.jpg"
            cv2.imwrite(str(path), vis)
            print(f"Saved {path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
