# Architecture

## Pipeline

```
source (image / video / dir)
        │
        ▼
  YOLO detector (domain-tuned rocket weights)
        │  boxes + scores
        ▼
  SORT tracker (default)          OR   Ultralytics ByteTrack (optional)
  Kalman + IoU + Hungarian
        │
        ▼
  annotated frames / video
```

## Packages

| Module | Role |
|--------|------|
| `detect.py` | Ultralytics YOLO predict → `Detection` list |
| `track_sort.py` | In-repo SORT (not `model.track()`) |
| `pipeline.py` | Frame iteration + tracker selection |
| `backends.py` | PyTorch CPU/CUDA, ONNX Runtime, TensorRT stub |
| `bench.py` | Warmup + timed loop → CSV / MD / PNG |
| `viz.py` | Box + stable ID overlay |
| `metrics.py` | mean / median / p95 / FPS helpers |

## Weights resolution

1. `weights/best.pt` (clone-friendly mirror)
2. `runs/train/rocket_detector/weights/best.pt` (local training output)

ONNX: `weights/best.onnx` or sibling of the `.pt` file.

## Dataset

Canonical YOLO layout at repo root (`data.yaml`, `train/`, `valid/`, `test/`). Training entrypoint: `python -m scripts.train`.
