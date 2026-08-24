"""Detection helpers wrapping Ultralytics YOLO and ONNX backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

PathLike = Union[str, Path]


@dataclass
class Detection:
    xyxy: tuple  # x1,y1,x2,y2
    score: float
    class_id: int = 0


def detections_to_array(dets: Sequence[Detection]) -> np.ndarray:
    if not dets:
        return np.empty((0, 6), dtype=np.float64)
    rows = [[*d.xyxy, d.score, d.class_id] for d in dets]
    return np.asarray(rows, dtype=np.float64)


def filter_detections(
    dets: Sequence[Detection],
    frame_shape: tuple[int, ...],
    max_area_frac: float = 0.04,
    min_area_frac: float = 0.0,
    max_det: int = 3,
) -> List[Detection]:
    """Drop absurd boxes (e.g. smoke trails) and keep top-scoring survivors.

    ``max_area_frac`` rejects boxes covering more than this fraction of the frame —
    plume/smoke FPs are typically huge vs a real rocket.
    """
    if not dets:
        return []
    h, w = int(frame_shape[0]), int(frame_shape[1])
    frame_area = max(float(h * w), 1.0)
    kept: List[Detection] = []
    for d in dets:
        x1, y1, x2, y2 = d.xyxy
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        frac = (bw * bh) / frame_area
        if max_area_frac > 0 and frac > max_area_frac:
            continue
        if min_area_frac > 0 and frac < min_area_frac:
            continue
        kept.append(d)
    kept.sort(key=lambda d: d.score, reverse=True)
    if max_det > 0:
        kept = kept[:max_det]
    return kept


class YOLODetector:
    """Thin Ultralytics wrapper used by the tracking pipeline."""

    def __init__(
        self,
        weights: PathLike,
        device: str = "cpu",
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        half: Optional[bool] = None,
        max_area_frac: float = 0.04,
        min_area_frac: float = 0.0,
        max_det: int = 3,
    ):
        from ultralytics import YOLO
        import torch

        self.weights = Path(weights)
        if not self.weights.exists():
            raise FileNotFoundError(
                f"Weights not found: {self.weights}\n"
                "Place best.pt under weights/ or runs/train/rocket_detector/weights/, "
                "or pass --weights explicitly. See README for Release / training notes."
            )
        if torch.cuda.is_available() and str(device) not in {"cpu", "CPU"}:
            torch.backends.cudnn.benchmark = True

        # .engine exports need an explicit task; .pt loads task from checkpoint.
        kw = {}
        if self.weights.suffix.lower() == ".engine":
            kw["task"] = "detect"
        self.model = YOLO(str(self.weights), **kw)
        try:
            self.model.fuse()
        except Exception:  # noqa: BLE001
            pass
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        if half is None:
            half = str(device) not in {"cpu", "CPU"}
        self.half = bool(half) and str(device) not in {"cpu", "CPU"}
        self.max_area_frac = float(max_area_frac)
        self.min_area_frac = float(min_area_frac)
        self.max_det = int(max_det)
        self._warmed = False

    def warmup(self, shape: tuple[int, int, int] = (720, 1280, 3)) -> None:
        """Run one dummy forward so later frames skip init overhead."""
        if self._warmed:
            return
        blank = np.zeros(shape, dtype=np.uint8)
        self.predict(blank)
        self._warmed = True

    def predict(self, image_bgr: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            source=image_bgr,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            half=self.half,
            verbose=False,
            max_det=max(self.max_det * 5, 10),  # over-fetch; we filter locally
        )
        out: List[Detection] = []
        if not results:
            return out
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return out
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = map(float, xyxy[i])
            out.append(Detection(xyxy=(x1, y1, x2, y2), score=float(conf[i]), class_id=int(cls[i])))
        return filter_detections(
            out,
            image_bgr.shape,
            max_area_frac=self.max_area_frac,
            min_area_frac=self.min_area_frac,
            max_det=self.max_det,
        )