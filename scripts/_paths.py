"""Shared path helpers for CLI scripts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def default_weights() -> Path:
    candidates = [
        ROOT / "weights" / "best.pt",
        ROOT / "runs" / "train" / "rocket_detector" / "weights" / "best.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def default_source() -> Path:
    for cand in [
        ROOT / "testing_media" / "testvid.mp4",
        ROOT / "assets" / "sample" / "demo_rocket.jpg",
        ROOT / "assets" / "sample",
        ROOT / "testing_media" / "IMG_0026.png",
    ]:
        if cand.exists():
            return cand
    return ROOT / "assets" / "sample"


def default_still() -> Path:
    """Prefer a known rocket still over incidental phone photos."""
    preferred = [
        ROOT / "assets" / "sample" / "demo_rocket.jpg",
        ROOT / "assets" / "results" / "demo_track_still.jpg",
    ]
    for cand in preferred:
        if cand.exists():
            return cand
    sample = ROOT / "assets" / "sample"
    if sample.exists():
        for p in sorted(sample.glob("*.jpg")):
            return p
        for p in sorted(sample.glob("*.png")):
            if p.name != "IMG_0026.png":
                return p
    return default_source()


def default_onnx() -> Path:
    candidates = [
        ROOT / "weights" / "best.onnx",
        ROOT / "runs" / "train" / "rocket_detector" / "weights" / "best.onnx",
        default_weights().with_suffix(".onnx"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def resolve_device(device: str) -> str:
    """Map CLI device aliases. ``auto`` picks CUDA when available."""
    device = (device or "cpu").strip().lower()
    if device in {"cuda", "gpu"}:
        return "0"
    if device == "auto":
        try:
            import torch

            return "0" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            return "cpu"
    return device
