"""Drawing helpers for annotated frames."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import cv2
import numpy as np

from .track_sort import TrackResult


def _color_for_id(track_id: int) -> tuple:
    rng = np.random.default_rng(track_id * 9973)
    c = rng.integers(64, 255, size=3)
    return int(c[0]), int(c[1]), int(c[2])


def draw_tracks(
    frame_bgr: np.ndarray,
    tracks: Sequence[TrackResult],
    class_names: Optional[Sequence[str]] = None,
    thickness: int = 2,
) -> np.ndarray:
    out = frame_bgr.copy()
    names = class_names or ["Rocket"]
    for t in tracks:
        x1, y1, x2, y2 = map(int, t.xyxy)
        color = _color_for_id(t.track_id)
        # Dashed-style thinner box for coasted (predicted) tracks
        box_thickness = 1 if t.coasted else thickness
        cv2.rectangle(out, (x1, y1), (x2, y2), color, box_thickness)
        label = f"ID {t.track_id}"
        if 0 <= t.class_id < len(names):
            label = f"{names[t.class_id]} {label}"
        if t.coasted:
            label += " ~"
        else:
            label += f" {t.score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            out,
            label,
            (x1 + 2, max(th + 2, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return out


def draw_detections(
    frame_bgr: np.ndarray,
    detections: Iterable,
    thickness: int = 2,
) -> np.ndarray:
    out = frame_bgr.copy()
    for d in detections:
        x1, y1, x2, y2 = map(int, d.xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), thickness)
        cv2.putText(
            out,
            f"{d.score:.2f}",
            (x1, max(15, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 220, 0),
            1,
            cv2.LINE_AA,
        )
    return out
