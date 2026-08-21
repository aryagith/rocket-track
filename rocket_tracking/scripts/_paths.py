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
        ROOT / "testing_media" / "IMG_0026.png",
        ROOT / "assets" / "sample",
    ]:
        if cand.exists():
            return cand
    return ROOT / "assets" / "sample"
