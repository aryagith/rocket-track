"""Fine-tune a YOLOv8n rocket detector for higher FPS.

Shorter than the full 200-epoch YOLOv8s recipe — enough to beat a COCO-pretrained
nano on launch footage while staying light for ~60 FPS targets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts._paths import ROOT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune YOLOv8n on the rocket dataset.")
    p.add_argument("--data", type=Path, default=ROOT / "data.yaml")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--device", default="0")
    p.add_argument("--name", default="rocket_detector_n")
    p.add_argument("--project", type=Path, default=ROOT / "runs" / "train")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--copy-to", type=Path, default=ROOT / "weights" / "best_n.pt")
    return p.parse_args()


def main() -> None:
    import multiprocessing
    import shutil

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
        device=args.device,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        patience=15,
        save=True,
        plots=True,
        amp=True,
        cache="disk",
        single_cls=True,
        verbose=True,
        seed=0,
        close_mosaic=5,
    )

    best = args.project / args.name / "weights" / "best.pt"
    print(f"Best weights: {best}")
    if best.exists():
        args.copy_to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, args.copy_to)
        print(f"Copied to {args.copy_to}")
    print("Speed check:")
    print(
        "  python -m scripts.run_track --source testing_media/rocket_launch.mov "
        "--weights weights/best_n.pt --device cuda --profile fast --half --no-save"
    )


if __name__ == "__main__":
    main()
