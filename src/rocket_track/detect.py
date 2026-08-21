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


class YOLODetector:
    """Thin Ultralytics wrapper used by the tracking pipeline."""

    def __init__(
        self,
        weights: PathLike,
        device: str = "cpu",
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
    ):
        from ultralytics import YOLO

        self.weights = Path(weights)
        if not self.weights.exists():
            raise FileNotFoundError(
                f"Weights not found: {self.weights}\n"
                "Place best.pt under weights/ or runs/train/rocket_detector/weights/, "
                "or pass --weights explicitly. See README for Release / training notes."
            )
        self.model = YOLO(str(self.weights))
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou

    def predict(self, image_bgr: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            source=image_bgr,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
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
        return out
