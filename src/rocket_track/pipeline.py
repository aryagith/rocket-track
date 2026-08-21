"""Detect + track pipeline over images / video."""

from __future__ import annotations

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
    ):
        self.tracker_name = tracker.lower()
        self.detector = YOLODetector(weights, device=device, imgsz=imgsz, conf=conf, iou=iou)
        self.sort = SortTracker(max_age=max_age, min_hits=min_hits, iou_threshold=track_iou)
        self._bt_model = None
        self._bt_cfg = "bytetrack.yaml"
        if self.tracker_name == "bytetrack":
            # Optional comparison path using Ultralytics built-in tracker.
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
    ) -> Path:
        source = Path(source)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.reset()

        writer = None
        out_path = out_dir / f"{source.stem}_tracked.mp4"
        frame_paths: List[Path] = []
        names = list(class_names) if class_names else ["Rocket"]

        for idx, frame in iter_frames(source):
            tracks = self.process_frame(frame)
            annotated = draw_tracks(frame, tracks, class_names=names)
            if is_image(source) or source.is_dir():
                fp = out_dir / f"{source.stem if is_image(source) else 'frame'}_{idx:06d}.jpg"
                if is_image(source):
                    fp = out_dir / f"{source.stem}_tracked.jpg"
                cv2.imwrite(str(fp), annotated)
                frame_paths.append(fp)
                if is_image(source):
                    return fp
            elif save_video:
                if writer is None:
                    h, w = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(out_path), fourcc, 30.0, (w, h))
                writer.write(annotated)

        if writer is not None:
            writer.release()
            return out_path
        if frame_paths:
            return frame_paths[0]
        raise RuntimeError(f"No frames processed from {source}")
