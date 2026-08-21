"""Inference backends: PyTorch (CPU/CUDA), ONNX Runtime, TensorRT (optional)."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from .detect import Detection

PathLike = Union[str, Path]


@dataclass
class BackendInfo:
    name: str
    available: bool
    reason: str = ""


def list_backend_availability() -> List[BackendInfo]:
    infos: List[BackendInfo] = []

    infos.append(BackendInfo("pytorch_cpu", True, "always"))

    cuda_ok = False
    try:
        import torch

        cuda_ok = bool(torch.cuda.is_available())
        infos.append(
            BackendInfo(
                "pytorch_cuda",
                cuda_ok,
                "CUDA available" if cuda_ok else "torch.cuda.is_available() is False",
            )
        )
    except Exception as exc:  # noqa: BLE001
        infos.append(BackendInfo("pytorch_cuda", False, f"torch import failed: {exc}"))

    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        infos.append(BackendInfo("onnx_cpu", "CPUExecutionProvider" in providers, str(providers)))
        infos.append(
            BackendInfo(
                "onnx_cuda",
                "CUDAExecutionProvider" in providers,
                "CUDA EP present" if "CUDAExecutionProvider" in providers else "CUDA EP not installed",
            )
        )
    except Exception as exc:  # noqa: BLE001
        infos.append(BackendInfo("onnx_cpu", False, f"onnxruntime missing: {exc}"))
        infos.append(BackendInfo("onnx_cuda", False, f"onnxruntime missing: {exc}"))

    trt_ok = False
    trt_reason = "TensorRT Python package / ultralytics engine export not available"
    try:
        import tensorrt  # noqa: F401

        trt_ok = True
        trt_reason = "tensorrt import ok (engine still required)"
    except Exception:
        pass
    infos.append(BackendInfo("tensorrt", trt_ok, trt_reason))
    return infos


def letterbox(
    image: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    h, w = image.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new_shape[0] - nh) // 2
    left = (new_shape[1] - nw) // 2
    out = np.full((new_shape[0], new_shape[1], 3), color, dtype=np.uint8)
    out[top : top + nh, left : left + nw] = resized
    return out, r, (left, top)


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thres]
    return keep


class InferenceBackend(ABC):
    name: str

    @abstractmethod
    def infer(self, image_bgr: np.ndarray) -> List[Detection]:
        raise NotImplementedError

    def timed_infer(self, image_bgr: np.ndarray) -> Tuple[List[Detection], float]:
        t0 = time.perf_counter()
        dets = self.infer(image_bgr)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return dets, dt_ms


class UltralyticsBackend(InferenceBackend):
    def __init__(
        self,
        weights: PathLike,
        device: str,
        name: str,
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
    ):
        from ultralytics import YOLO

        self.name = name
        self.model = YOLO(str(weights))
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou

    def infer(self, image_bgr: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            source=image_bgr,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        out: List[Detection] = []
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return out
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        for i in range(len(xyxy)):
            out.append(
                Detection(
                    xyxy=tuple(map(float, xyxy[i])),
                    score=float(conf[i]),
                    class_id=int(cls[i]),
                )
            )
        return out


class ONNXBackend(InferenceBackend):
    """YOLOv8 ONNX via onnxruntime (CPU or CUDA EP)."""

    def __init__(
        self,
        onnx_path: PathLike,
        providers: Sequence[str],
        name: str,
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
    ):
        import onnxruntime as ort

        self.name = name
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.session = ort.InferenceSession(str(onnx_path), providers=list(providers))
        self.input_name = self.session.get_inputs()[0].name

    def infer(self, image_bgr: np.ndarray) -> List[Detection]:
        img, ratio, (pad_w, pad_h) = letterbox(image_bgr, (self.imgsz, self.imgsz))
        blob = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...]
        outputs = self.session.run(None, {self.input_name: blob})
        pred = np.squeeze(outputs[0])
        # Ultralytics ONNX: (4+nc, N) or (N, 4+nc)
        if pred.ndim != 2:
            return []
        if pred.shape[0] < pred.shape[1] and pred.shape[0] <= 84:
            pred = pred.T
        boxes = pred[:, :4]
        scores_all = pred[:, 4:]
        if scores_all.shape[1] == 0:
            return []
        class_ids = np.argmax(scores_all, axis=1)
        scores = scores_all[np.arange(scores_all.shape[0]), class_ids]
        mask = scores >= self.conf
        boxes, scores, class_ids = boxes[mask], scores[mask], class_ids[mask]
        if boxes.size == 0:
            return []

        # xywh (center) -> xyxy in letterboxed space
        xyxy = np.zeros_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

        keep = nms_xyxy(xyxy, scores, self.iou)
        out: List[Detection] = []
        for i in keep:
            x1 = (xyxy[i, 0] - pad_w) / ratio
            y1 = (xyxy[i, 1] - pad_h) / ratio
            x2 = (xyxy[i, 2] - pad_w) / ratio
            y2 = (xyxy[i, 3] - pad_h) / ratio
            out.append(
                Detection(
                    xyxy=(float(x1), float(y1), float(x2), float(y2)),
                    score=float(scores[i]),
                    class_id=int(class_ids[i]),
                )
            )
        return out


def resolve_weights(weights: PathLike, prefer_onnx: bool = False) -> Path:
    p = Path(weights)
    if prefer_onnx and p.suffix.lower() == ".pt":
        cand = p.with_suffix(".onnx")
        if cand.exists():
            return cand
    return p


def create_backend(
    backend: str,
    weights: PathLike,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.45,
    onnx_path: Optional[PathLike] = None,
) -> InferenceBackend:
    backend = backend.lower().strip()
    weights = Path(weights)

    if backend == "pytorch_cpu":
        return UltralyticsBackend(weights, "cpu", "pytorch_cpu", imgsz, conf, iou)
    if backend == "pytorch_cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("pytorch_cuda requested but CUDA is unavailable")
        return UltralyticsBackend(weights, "0", "pytorch_cuda", imgsz, conf, iou)

    if backend in {"onnx", "onnx_cpu", "onnx_cuda"}:
        onnx_file = Path(onnx_path) if onnx_path else resolve_weights(weights, prefer_onnx=True)
        if onnx_file.suffix.lower() != ".onnx":
            # try sibling / weights/best.onnx
            candidates = [
                weights.with_suffix(".onnx"),
                Path("weights/best.onnx"),
                Path("runs/train/rocket_detector/weights/best.onnx"),
            ]
            onnx_file = next((c for c in candidates if c.exists()), onnx_file)
        if not onnx_file.exists():
            raise FileNotFoundError(
                f"ONNX model not found near {weights}. Run: python -m scripts.export_onnx"
            )
        if backend == "onnx_cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            name = "onnx_cuda"
        else:
            providers = ["CPUExecutionProvider"]
            name = "onnx_cpu" if backend != "onnx" else "onnx_cpu"
        return ONNXBackend(onnx_file, providers, name, imgsz, conf, iou)

    if backend == "tensorrt":
        raise RuntimeError(
            "TensorRT backend skipped: no packaged TensorRT engine in this repo. "
            "Export with ultralytics `format=engine` on a machine with TensorRT, then add a loader."
        )

    raise ValueError(f"Unknown backend: {backend}")


def peak_vram_mb() -> Optional[float]:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    except Exception:  # noqa: BLE001
        return None


def reset_vram_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
