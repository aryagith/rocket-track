"""Train the rocket YOLOv8s detector (RTX 4060-oriented defaults).

Preserves the original training recipe from legacy/train.py and points at
repo-root data.yaml by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts._paths import ROOT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train rocket YOLOv8s detector.")
    p.add_argument("--data", type=Path, default=ROOT / "data.yaml")
    p.add_argument("--model", default="yolov8s.pt")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--device", default="0")
    p.add_argument("--name", default="rocket_detector")
    p.add_argument("--project", type=Path, default=ROOT / "runs" / "train")
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


def main() -> None:
    import multiprocessing

    multiprocessing.freeze_support()
    args = parse_args()
    import torch
    from ultralytics import YOLO

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        project=str(args.project),
        workers=args.workers,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15,
        translate=0.1,
        scale=0.5,
        shear=2,
        perspective=0.0001,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.3,
        auto_augment="randaugment",
        erasing=0.4,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        iou=0.7,
        patience=50,
        save=True,
        save_period=10,
        cache="disk",
        device=args.device,
        val=True,
        plots=True,
        amp=True,
        fraction=1.0,
        verbose=True,
        seed=0,
        deterministic=False,
        single_cls=True,
        close_mosaic=10,
    )

    best = args.project / args.name / "weights" / "best.pt"
    print(f"Best weights: {best}")
    print("Optional ONNX export: python -m scripts.export_onnx")


if __name__ == "__main__":
    main()
