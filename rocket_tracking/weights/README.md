# Weights

| File | Purpose |
|------|---------|
| `best.pt` | Default YOLOv8s rocket detector (product weights) |
| `best.onnx` | Same model for ONNX Runtime benches |

These are mirrored from `runs/train/rocket_detector/weights/` after training.

If missing in a clone:

```bash
# After you obtain best.pt (Release asset or local train output):
python -m scripts.export_onnx --weights weights/best.pt --out weights/best.onnx
```

Do not use root `yolov8s.pt` / `yolov8n.pt` as the product detector — those are generic Ultralytics bases.
