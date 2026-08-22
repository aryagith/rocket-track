"""Detect + track pipeline over images / video."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from .detect import YOLODetector, detections_to_array
from .track_sort import SortTracker, TrackResult
from .viz import draw_tracks

PathLike = Union[str, Path]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTS


def probe_video_fps(source: Path, default: float = 30.0) -> float:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        return default
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    if fps <= 1e-3 or fps > 240:
        return default
    return fps


def iter_frames(source: PathLike) -> Iterator[Tuple[int, np.ndarray]]:
    source = Path(source)
    if source.is_dir():
        files = sorted(
            [p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS],
            key=lambda p: p.name,
        )
        for i, p in enumerate(files):
            img = cv2.imread(str(p))
            if img is not None:
                yield i, img
        return

    if is_image(source):
        img = cv2.imread(str(source))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {source}")
        yield 0, img
        return

    if is_video(source) or source.exists():
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video/source: {source}")
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
        cap.release()
        return

    raise FileNotFoundError(f"Source not found: {source}")


@dataclass
class RunStats:
    out_path: Path
    n_frames: int
    elapsed_s: float
    fps: float
    infer_elapsed_s: float
    infer_fps: float
    frames_with_tracks: int


class TrackPipeline:
    def __init__(
        self,
        weights: PathLike,
        device: str = "cpu",
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        tracker: str = "sort",
        max_age: int = 30,
        min_hits: int = 3,
        track_iou: float = 0.3,
        half: Optional[bool] = None,
    ):
        self.tracker_name = tracker.lower()
        self.detector = YOLODetector(
            weights, device=device, imgsz=imgsz, conf=conf, iou=iou, half=half
        )
        try:
            self.detector.warmup()
        except Exception:  # noqa: BLE001
            pass
        self.sort = SortTracker(max_age=max_age, min_hits=min_hits, iou_threshold=track_iou)
        self._bt_model = None
        self._bt_cfg = "bytetrack.yaml"
        if self.tracker_name == "bytetrack":
            self._bt_model = self.detector.model
            repo_cfg = Path(__file__).resolve().parents[2] / "configs" / "bytetrack.yaml"
            if repo_cfg.exists():
                self._bt_cfg = str(repo_cfg)

    def reset(self) -> None:
        self.sort.reset()

    def process_frame(self, frame_bgr: np.ndarray) -> List[TrackResult]:
        if self.tracker_name == "bytetrack":
            results = self._bt_model.track(
                source=frame_bgr,
                persist=True,
                tracker=self._bt_cfg,
                conf=self.detector.conf,
                iou=self.detector.iou,
                imgsz=self.detector.imgsz,
                device=self.detector.device,
                half=self.detector.half,
                verbose=False,
            )
            tracks: List[TrackResult] = []
            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                return tracks
            ids = boxes.id
            if ids is None:
                return tracks
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            tid = ids.cpu().numpy().astype(int)
            for i in range(len(xyxy)):
                tracks.append(
                    TrackResult(
                        track_id=int(tid[i]),
                        xyxy=tuple(map(float, xyxy[i])),
                        score=float(conf[i]),
                        class_id=int(cls[i]),
                    )
                )
            return tracks

        dets = self.detector.predict(frame_bgr)
        return self.sort.update(detections_to_array(dets))

    def run(
        self,
        source: PathLike,
        out_dir: PathLike,
        save_video: bool = True,
        class_names: Optional[Sequence[str]] = None,
        warmup: int = 10,
    ) -> RunStats:
        source = Path(source)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.reset()

        writer = None
        safe_stem = source.stem.replace(" ", "_")
        out_path = out_dir / f"{safe_stem}_tracked.mp4"
        frame_paths: List[Path] = []
        names = list(class_names) if class_names else ["Rocket"]
        video_fps = probe_video_fps(source) if is_video(source) else 30.0

        n_frames = 0
        frames_with_tracks = 0
        infer_elapsed = 0.0
        t0 = time.perf_counter()

        for idx, frame in iter_frames(source):
            t_infer0 = time.perf_counter()
            tracks = self.process_frame(frame)
            # Discard warmup from infer timing only
            if idx >= warmup:
                infer_elapsed += time.perf_counter() - t_infer0
            n_frames += 1
            if tracks:
                frames_with_tracks += 1

            # Speed path: skip draw/encode when not saving
            if not save_video and not is_image(source) and not source.is_dir():
                continue

            annotated = draw_tracks(frame, tracks, class_names=names)
            if is_image(source) or source.is_dir():
                if is_image(source):
                    fp = out_dir / f"{safe_stem}_tracked.jpg"
                else:
                    fp = out_dir / f"frame_{idx:06d}.jpg"
                cv2.imwrite(str(fp), annotated)
                frame_paths.append(fp)
                if is_image(source):
                    elapsed = time.perf_counter() - t0
                    return RunStats(
                        out_path=fp,
                        n_frames=1,
                        elapsed_s=elapsed,
                        fps=(1.0 / elapsed) if elapsed > 0 else float("nan"),
                        infer_elapsed_s=elapsed,
                        infer_fps=(1.0 / elapsed) if elapsed > 0 else float("nan"),
                        frames_with_tracks=frames_with_tracks,
                    )
            elif save_video:
                if writer is None:
                    h, w = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(out_path), fourcc, video_fps, (w, h))
                writer.write(annotated)

        elapsed = time.perf_counter() - t0
        fps = (n_frames / elapsed) if elapsed > 0 else float("nan")
        timed_frames = max(0, n_frames - warmup)
        infer_fps = (timed_frames / infer_elapsed) if infer_elapsed > 0 else float("nan")

        if writer is not None:
            writer.release()
            return RunStats(
                out_path, n_frames, elapsed, fps, infer_elapsed, infer_fps, frames_with_tracks
            )
        if frame_paths:
            return RunStats(
                frame_paths[0], n_frames, elapsed, fps, infer_elapsed, infer_fps, frames_with_tracks
            )
        if not save_video:
            return RunStats(
                out_dir / "(no file — --no-save)",
                n_frames,
                elapsed,
                fps,
                infer_elapsed,
                infer_fps,
                frames_with_tracks,
            )
        raise RuntimeError(f"No frames processed from {source}")
