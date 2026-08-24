"""Live webcam rocket track for a phone showing a rocket photo (move the phone).

Default ``--mode prop`` is tuned for that: lower conf, larger allowed box size
(the launch-video smoke filter was rejecting the phone), fast track lock.

Hold the phone so the rocket image is sharp and fills a good chunk of the
camera view (avoid glare / extreme tilt).

Keys:
  q / ESC  quit
  r        reset tracker IDs
  s        save a still under outputs/webcam/
  [/]      conf -0.05 / +0.05
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from scripts._paths import ROOT, default_weights, resolve_device
from rocket_track.pipeline import TrackPipeline
from rocket_track.viz import draw_tracks

# prop = phone with a rocket picture; launch = smoke-filter settings for real footage
MODES = {
    "prop": {
        "imgsz": 640,
        "conf": 0.15,
        "coast_frames": 20,
        "max_area_frac": 0.35,  # phone can be large in-frame; don't treat as smoke
        "min_area_frac": 0.0005,
        "max_det": 1,
        "min_hits": 1,
        "track_iou": 0.2,
    },
    "launch": {
        "imgsz": 512,
        "conf": 0.30,
        "coast_frames": 15,
        "max_area_frac": 0.04,
        "min_area_frac": 0.0,
        "max_det": 2,
        "min_hits": 3,
        "track_iou": 0.3,
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Live webcam detect+SORT — default: track a phone showing a rocket picture."
    )
    p.add_argument("--camera", type=int, default=0, help="OpenCV camera index (0=default webcam)")
    p.add_argument("--weights", type=Path, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--mode",
        choices=sorted(MODES.keys()),
        default="prop",
        help="prop=phone photo (default); launch=precision settings for sky/smoke footage",
    )
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--coast-frames", type=int, default=None)
    p.add_argument("--max-area-frac", type=float, default=None)
    p.add_argument("--min-area-frac", type=float, default=None)
    p.add_argument("--max-det", type=int, default=None)
    p.add_argument("--min-hits", type=int, default=None)
    p.add_argument("--width", type=int, default=1280, help="Request capture width (0=leave default)")
    p.add_argument("--height", type=int, default=720, help="Request capture height (0=leave default)")
    p.add_argument(
        "--mirror",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mirror preview horizontally (default on)",
    )
    return p.parse_args()


def _pick(cli_val, mode_val):
    return mode_val if cli_val is None else cli_val


def main() -> None:
    args = parse_args()
    mode = MODES[args.mode]
    device = resolve_device(args.device)
    weights = args.weights or default_weights(prefer_nano=False)

    imgsz = _pick(args.imgsz, mode["imgsz"])
    conf = float(_pick(args.conf, mode["conf"]))
    coast = int(_pick(args.coast_frames, mode["coast_frames"]))
    max_area = float(_pick(args.max_area_frac, mode["max_area_frac"]))
    min_area = float(_pick(args.min_area_frac, mode["min_area_frac"]))
    max_det = int(_pick(args.max_det, mode["max_det"]))
    min_hits = int(_pick(args.min_hits, mode["min_hits"]))
    track_iou = float(mode["track_iou"])

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")

    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    def make_pipe(conf_val: float) -> TrackPipeline:
        return TrackPipeline(
            weights=weights,
            device=device,
            imgsz=imgsz,
            conf=conf_val,
            half=device != "cpu",
            tracker="sort",
            coast_frames=coast,
            max_area_frac=max_area,
            min_area_frac=min_area,
            max_det=max_det,
            min_hits=min_hits,
            track_iou=track_iou,
        )

    pipe = make_pipe(conf)

    print(
        f"mode={args.mode}  webcam={args.camera}  weights={weights}  device={device}  "
        f"imgsz={imgsz}  conf={conf}  max_area_frac={max_area}  max_det={max_det}  min_hits={min_hits}"
    )
    print("Show a clear rocket photo on the phone; move slowly at first.  q=quit r=reset s=save [/]=conf")

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

        hud = (
            f"{args.mode}  FPS {fps:.1f}  conf {conf:.2f}  tracks {len(tracks)}  "
            f"[q]uit [r]eset [s]ave [/] conf"
        )
        cv2.putText(vis, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(vis, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 1, cv2.LINE_AA)
        tip = "Fill the view with the rocket photo; reduce glare"
        cv2.putText(vis, tip, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(vis, tip, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 1, cv2.LINE_AA)
        cv2.imshow(win, vis)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            pipe.reset()
            print("tracker reset")
        if key == ord("s"):
            path = still_dir / f"still_{int(time.time())}.jpg"
            cv2.imwrite(str(path), vis)
            print(f"Saved {path}")
        if key == ord("["):
            conf = max(0.05, round(conf - 0.05, 2))
            pipe = make_pipe(conf)
            print(f"conf={conf}")
        if key == ord("]"):
            conf = min(0.90, round(conf + 0.05, 2))
            pipe = make_pipe(conf)
            print(f"conf={conf}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
