"""Single-target lock tracker: acquire, hold, coast, re-acquire."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pytest

from rocket_track.lock_track import LockPhase, LockTracker

FRAME_W, FRAME_H = 1280, 720
FPS = 30.0
DT = 1.0 / FPS

# Rocket climbing through frame centre.
START = (640.0, 600.0)
VEL_Y = -80.0  # px/s
BOX_W, BOX_H = 40.0, 90.0


def box_at(t: float, cx: Optional[float] = None) -> Tuple[float, float, float, float]:
    cy = START[1] + VEL_Y * t
    cx = START[0] if cx is None else cx
    return (cx - BOX_W / 2, cy - BOX_H / 2, cx + BOX_W / 2, cy + BOX_H / 2)


def dets(*boxes: Sequence[float], score: float = 0.9) -> np.ndarray:
    if not boxes:
        return np.empty((0, 6), dtype=np.float64)
    return np.array([[*b, score, 0] for b in boxes], dtype=np.float64)


def make_tracker(**kw) -> LockTracker:
    params = dict(frame_size=(FRAME_W, FRAME_H), confirm_hits=3, coast_s=0.5)
    params.update(kw)
    return LockTracker(**params)


def acquire(tracker: LockTracker, t0: float = 0.0, frames: int = 3) -> float:
    """Feed consistent detections until the tracker locks. Returns last timestamp."""
    t = t0
    for _ in range(frames):
        tracker.update(dets(box_at(t)), t)
        t += DT
    return t - DT


def test_does_not_emit_before_confirmation():
    tracker = make_tracker()

    first = tracker.update(dets(box_at(0.0)), 0.0)
    assert first.phase is LockPhase.ACQUIRING
    assert first.xyxy is None

    second = tracker.update(dets(box_at(DT)), DT)
    assert second.phase is LockPhase.ACQUIRING
    assert second.xyxy is None


def test_locks_after_confirm_hits():
    tracker = make_tracker()
    t = 0.0
    for _ in range(2):
        tracker.update(dets(box_at(t)), t)
        t += DT

    state = tracker.update(dets(box_at(t)), t)

    assert state.phase is LockPhase.LOCKED
    assert state.xyxy is not None
    assert state.coasted is False


def test_reports_pixel_error_from_frame_centre():
    tracker = make_tracker()
    t = 0.0
    # Hold the target still, offset right and below frame centre.
    offset_box = (740.0 - BOX_W / 2, 460.0 - BOX_H / 2, 740.0 + BOX_W / 2, 460.0 + BOX_H / 2)
    state = None
    for _ in range(6):
        state = tracker.update(dets(offset_box), t)
        t += DT

    dx, dy = state.pixel_error
    assert abs(dx - 100.0) < 5.0
    assert abs(dy - 100.0) < 5.0


def test_coasts_through_detector_dropout():
    tracker = make_tracker()
    t = acquire(tracker) + DT

    state = tracker.update(dets(), t)

    assert state.phase is LockPhase.COASTING
    assert state.coasted is True
    assert state.xyxy is not None


def test_drops_lock_after_coast_window():
    tracker = make_tracker(coast_s=0.2)
    t = acquire(tracker)

    state = None
    for _ in range(10):  # 10 frames = 0.33 s > 0.2 s coast window
        t += DT
        state = tracker.update(dets(), t)

    assert state.phase is LockPhase.ACQUIRING
    assert state.xyxy is None


def test_carries_last_accepted_detection_score_through_coasting():
    tracker = make_tracker()
    t = 0.0
    state = None
    for _ in range(3):
        state = tracker.update(dets(box_at(t), score=0.77), t)
        t += DT
    assert state.score == pytest.approx(0.77)

    t += DT
    coasting = tracker.update(dets(), t)
    assert coasting.coasted is True
    assert coasting.score == pytest.approx(0.77)


def test_smoke_box_does_not_steal_lock():
    """A big, high-confidence plume elsewhere must not capture the tracker."""
    tracker = make_tracker()
    t = acquire(tracker)

    smoke = (100.0, 560.0, 400.0, 700.0)
    state = None
    for _ in range(5):
        t += DT
        rocket = box_at(t)
        state = tracker.update(
            np.array(
                [[*rocket, 0.90, 0], [*smoke, 0.99, 0]],
                dtype=np.float64,
            ),
            t,
        )

    assert state.center is not None
    assert abs(state.center[0] - START[0]) < 25.0


def test_rejects_detection_with_implausible_size():
    """Right place, wrong scale: a box many times the target size is not it."""
    tracker = make_tracker()
    t = acquire(tracker)

    t += DT
    cx = START[0]
    cy = START[1] + VEL_Y * t
    oversized = (cx - 5 * BOX_W, cy - 5 * BOX_H, cx + 5 * BOX_W, cy + 5 * BOX_H)
    state = tracker.update(dets(oversized), t)

    assert state.phase is LockPhase.COASTING
    assert state.coasted is True


def test_lead_prediction_is_ahead_of_current_estimate():
    tracker = make_tracker(lead_s=0.1)
    t = 0.0
    state = None
    for _ in range(8):
        state = tracker.update(dets(box_at(t)), t)
        t += DT

    assert state.velocity[1] < -40.0  # climbing
    lead_cy = (state.lead_xyxy[1] + state.lead_xyxy[3]) / 2.0
    assert lead_cy < state.center[1] - 4.0


def test_reacquires_after_drop():
    tracker = make_tracker(coast_s=0.2)
    t = acquire(tracker)
    for _ in range(10):
        t += DT
        tracker.update(dets(), t)

    # A new target appears elsewhere in frame.
    state = None
    for _ in range(3):
        t += DT
        state = tracker.update(dets(box_at(t, cx=300.0)), t)

    assert state.phase is LockPhase.LOCKED
    assert state.xyxy is not None
