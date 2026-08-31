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


LOCK_COLOR = (80, 230, 80)
COAST_COLOR = (60, 190, 255)
IDLE_COLOR = (150, 150, 150)
LEAD_COLOR = (240, 200, 90)


def draw_lock_state(
    frame_bgr: np.ndarray,
    state,
    show_lead: bool = True,
    thickness: int = 2,
) -> np.ndarray:
    """Overlay the single-target lock and the error a pan-tilt loop would null.

    The line from frame centre to target centre is the control signal: a gimbal
    steers to shrink it. Colour encodes trust — green when a detection backed
    this frame, amber while coasting on prediction alone.
    """
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    fcx, fcy = w // 2, h // 2

    cv2.drawMarker(out, (fcx, fcy), IDLE_COLOR, cv2.MARKER_CROSS, 14, 1)

    if state.xyxy is None:
        cv2.putText(
            out, "ACQUIRING", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, IDLE_COLOR, 1, cv2.LINE_AA
        )
        return out

    color = COAST_COLOR if state.coasted else LOCK_COLOR
    x1, y1, x2, y2 = (int(v) for v in state.xyxy)
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 1 if state.coasted else thickness)

    cx, cy = int(state.center[0]), int(state.center[1])
    cv2.drawMarker(out, (cx, cy), color, cv2.MARKER_CROSS, 18, 1)
    cv2.line(out, (fcx, fcy), (cx, cy), color, 1, cv2.LINE_AA)

    if show_lead and state.lead_xyxy is not None:
        lx1, ly1, lx2, ly2 = (int(v) for v in state.lead_xyxy)
        if (lx1, ly1, lx2, ly2) != (x1, y1, x2, y2):
            cv2.rectangle(out, (lx1, ly1), (lx2, ly2), LEAD_COLOR, 1)

    dx, dy = state.pixel_error
    vx, vy = state.velocity
    lines = [
        f"{state.phase.value.upper()}{' ~' if state.coasted else f' {state.score:.2f}'}",
        f"err {dx:+.0f},{dy:+.0f} px",
        f"vel {vx:+.0f},{vy:+.0f} px/s",
        f"sigma {state.uncertainty_px:.1f} px",
    ]
    for i, text in enumerate(lines):
        cv2.putText(
            out,
            text,
            (8, 22 + i * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
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
