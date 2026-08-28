"""Single-target lock tracker for one launching vehicle.

Launches in this domain have one rocket, so there is no identity to maintain:
the tracker acquires a target, holds it, coasts through detector dropouts, and
re-acquires from scratch once the lock is lost. Output is shaped for a pan-tilt
loop rather than for drawing IDs — pixel error from frame centre, velocity, and
a health signal the controller can use to decide whether to slew at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence, Tuple

import numpy as np

from .kalman import (
    CX,
    CY,
    BoxMeasurement,
    ConstantAccelerationBox,
    TrackFilter,
    box_to_measurement,
    measurement_to_box,
)

Box = Tuple[float, float, float, float]


class LockPhase(Enum):
    """Acquiring emits nothing; locked and coasting both emit a box."""

    ACQUIRING = "acquiring"
    LOCKED = "locked"
    COASTING = "coasting"


@dataclass(frozen=True)
class LockState:
    """One frame of tracker output, sized for a control loop."""

    phase: LockPhase
    xyxy: Optional[Box] = None
    center: Optional[Tuple[float, float]] = None
    pixel_error: Optional[Tuple[float, float]] = None
    velocity: Optional[Tuple[float, float]] = None
    coasted: bool = False
    frames_since_update: int = 0
    uncertainty_px: float = float("inf")
    lead_xyxy: Optional[Box] = None
    score: float = 0.0

    @property
    def has_target(self) -> bool:
        return self.xyxy is not None


def _as_detection_array(detections) -> np.ndarray:
    if detections is None:
        return np.empty((0, 6), dtype=np.float64)
    arr = np.asarray(detections, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 6), dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


class LockTracker:
    """Acquire one target, hold it, coast briefly, then re-acquire."""

    def __init__(
        self,
        frame_size: Tuple[int, int],
        confirm_hits: int = 3,
        coast_s: float = 0.5,
        lead_s: float = 0.0,
        gate_chi2: float = 9.21,
        size_ratio: Tuple[float, float] = (0.5, 2.0),
        motion: Optional[ConstantAccelerationBox] = None,
        measurement: Optional[BoxMeasurement] = None,
    ):
        self.frame_w, self.frame_h = int(frame_size[0]), int(frame_size[1])
        self.confirm_hits = int(confirm_hits)
        self.coast_s = float(coast_s)
        self.lead_s = float(lead_s)
        self.gate_chi2 = float(gate_chi2)
        self.size_ratio = (float(size_ratio[0]), float(size_ratio[1]))
        self._motion = motion or ConstantAccelerationBox()
        self._measurement = measurement or BoxMeasurement()
        self.reset()

    def reset(self) -> None:
        self._filter: Optional[TrackFilter] = None
        self._phase = LockPhase.ACQUIRING
        self._hits = 0
        self._last_update_t = 0.0
        self._frames_since_update = 0
        self._last_score = 0.0

    @property
    def phase(self) -> LockPhase:
        return self._phase

    def update(self, detections, t: float) -> LockState:
        """Advance to time ``t`` with this frame's detections."""
        dets = _as_detection_array(detections)
        t = float(t)

        if self._filter is None:
            return self._try_start(dets, t)

        self._filter.predict_to(t)
        if self._filter.diverged:
            self._drop()
            return self._try_start(dets, t)

        chosen = self._select(dets)
        if chosen is not None:
            self._filter.update(box_to_measurement(chosen))
            if self._filter.diverged:
                self._drop()
                return self._try_start(dets, t)
            self._hits += 1
            self._last_score = float(chosen[4]) if len(chosen) > 4 else 1.0
            self._last_update_t = t
            self._frames_since_update = 0
            self._phase = (
                LockPhase.LOCKED if self._hits >= self.confirm_hits else LockPhase.ACQUIRING
            )
            return self._emit_for_phase()

        self._frames_since_update += 1
        if self._phase is LockPhase.ACQUIRING:
            # Unconfirmed candidates do not get to coast.
            self._drop()
            return self._acquiring_state()

        if (t - self._last_update_t) > self.coast_s:
            self._drop()
            return self._acquiring_state()

        self._phase = LockPhase.COASTING
        return self._emit()

    def _try_start(self, dets: np.ndarray, t: float) -> LockState:
        best = self._best(dets)
        if best is None:
            return self._acquiring_state()
        self._filter = TrackFilter(self._motion, self._measurement)
        self._filter.initialize(box_to_measurement(best), t)
        self._hits = 1
        self._last_score = float(best[4]) if best.shape[0] > 4 else 1.0
        self._last_update_t = t
        self._frames_since_update = 0
        self._phase = (
            LockPhase.LOCKED if self._hits >= self.confirm_hits else LockPhase.ACQUIRING
        )
        return self._emit_for_phase()

    def _select(self, dets: np.ndarray) -> Optional[Sequence[float]]:
        """Pick the detection most consistent with the current estimate.

        Two gates must both pass. The chi-square test on the centre innovation
        scales with the filter's own covariance, so it is tight after a good
        update and forgiving after a dropout. The size band rejects boxes at
        plausible positions but implausible scale, which is how a plume that
        engulfs the vehicle gets in. Among survivors the most consistent wins,
        not the most confident: a high-scoring plume must not outrank the rocket.
        """
        f = self._filter
        if f is None or dets.shape[0] == 0:
            return None

        lo, hi = self.size_ratio
        w_est, h_est = f.size
        best: Optional[np.ndarray] = None
        best_d2 = float("inf")
        for det in dets:
            z = box_to_measurement(det)
            w, h = z[2], z[3]
            if w <= 0.0 or h <= 0.0:
                continue
            r_w = w / max(w_est, 1e-6)
            r_h = h / max(h_est, 1e-6)
            if not (lo <= r_w <= hi and lo <= r_h <= hi):
                continue
            d2 = f.center_mahalanobis2(z)
            if d2 <= self.gate_chi2 and d2 < best_d2:
                best, best_d2 = det, d2
        return best

    @staticmethod
    def _best(dets: np.ndarray) -> Optional[np.ndarray]:
        if dets.shape[0] == 0:
            return None
        scores = dets[:, 4] if dets.shape[1] > 4 else np.ones(dets.shape[0])
        return dets[int(np.argmax(scores))]

    def _drop(self) -> None:
        self._filter = None
        self._phase = LockPhase.ACQUIRING
        self._hits = 0

    def _emit_for_phase(self) -> LockState:
        """Only confirmed tracks are published; candidates stay invisible."""
        if self._phase is LockPhase.ACQUIRING:
            return self._acquiring_state()
        return self._emit()

    def _acquiring_state(self) -> LockState:
        return LockState(
            phase=LockPhase.ACQUIRING,
            frames_since_update=self._frames_since_update,
        )

    def _emit(self) -> LockState:
        f = self._filter
        assert f is not None
        cx, cy = f.center
        w, h = f.size
        box = measurement_to_box((cx, cy, w, h))
        lead = f.state_at(self.lead_s)
        lead_box = measurement_to_box((lead[CX], lead[CY], w, h))
        uncertainty = float(np.sqrt(max(f.P[CX, CX] + f.P[CY, CY], 0.0)))
        return LockState(
            phase=self._phase,
            xyxy=box,
            center=(cx, cy),
            pixel_error=(cx - self.frame_w / 2.0, cy - self.frame_h / 2.0),
            velocity=f.velocity,
            coasted=self._phase is LockPhase.COASTING,
            frames_since_update=self._frames_since_update,
            uncertainty_px=uncertainty,
            lead_xyxy=lead_box,
            score=self._last_score,
        )
