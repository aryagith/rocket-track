"""Linear Kalman core over a constant-acceleration box in pixel space."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from rocket_track.kalman import (
    BoxMeasurement,
    ConstantAccelerationBox,
    TrackFilter,
    box_to_measurement,
)

# Rocket-ish ascent: rising fast in -y, accelerating, box roughly constant.
X0 = (640.0, 900.0)
V0 = (5.0, -120.0)
ACC = (0.0, -90.0)
SIZE = (40.0, 90.0)


def truth_at(t: float) -> Tuple[float, float]:
    cx = X0[0] + V0[0] * t + 0.5 * ACC[0] * t * t
    cy = X0[1] + V0[1] * t + 0.5 * ACC[1] * t * t
    return cx, cy


def simulate(dts: Sequence[float], noise_px: float = 2.0, seed: int = 0):
    """Yield (timestamp, noisy measurement) pairs along the true trajectory."""
    rng = np.random.default_rng(seed)
    out: List[Tuple[float, np.ndarray]] = []
    t = 0.0
    for dt in dts:
        cx, cy = truth_at(t)
        z = np.array(
            [
                cx + rng.normal(0.0, noise_px),
                cy + rng.normal(0.0, noise_px),
                SIZE[0] + rng.normal(0.0, noise_px * 0.5),
                SIZE[1] + rng.normal(0.0, noise_px * 0.5),
            ],
            dtype=np.float64,
        )
        out.append((t, z))
        t += dt
    return out


def build_filter() -> TrackFilter:
    return TrackFilter(motion=ConstantAccelerationBox(), measurement=BoxMeasurement())


def run(samples) -> TrackFilter:
    f = build_filter()
    t0, z0 = samples[0]
    f.initialize(z0, t0)
    for t, z in samples[1:]:
        f.predict_to(t)
        f.update(z)
    return f


def test_tracks_constant_acceleration_trajectory():
    samples = simulate([1.0 / 30.0] * 45)
    f = run(samples)

    t_end = samples[-1][0]
    cx, cy = truth_at(t_end)
    est_cx, est_cy = f.center
    assert abs(est_cx - cx) < 5.0
    assert abs(est_cy - cy) < 5.0

    true_vy = V0[1] + ACC[1] * t_end
    assert abs(f.velocity[1] - true_vy) < 0.15 * abs(true_vy)


def test_predict_uses_elapsed_seconds_not_frame_count():
    rng = np.random.default_rng(7)
    dts = list(rng.uniform(0.01, 0.09, size=60))
    f = run(simulate(dts, seed=3))

    t_end = sum(dts[:-1])
    cx, cy = truth_at(t_end)
    est_cx, est_cy = f.center
    assert abs(est_cx - cx) < 6.0
    assert abs(est_cy - cy) < 6.0


def test_box_to_measurement_round_trips():
    z = box_to_measurement((100.0, 200.0, 140.0, 290.0))
    assert np.allclose(z, [120.0, 245.0, 40.0, 90.0])


def offset_measurement(f: TrackFilter, dx: float, dy: float) -> np.ndarray:
    """A detection sitting ``(dx, dy)`` px away from the current prediction."""
    cx, cy = f.center
    w, h = f.size
    return np.array([cx + dx, cy + dy, w, h], dtype=np.float64)


def test_center_mahalanobis2_separates_consistent_from_outlier():
    f = run(simulate([1.0 / 30.0] * 30))

    assert f.center_mahalanobis2(offset_measurement(f, 1.0, 1.0)) < 9.21
    assert f.center_mahalanobis2(offset_measurement(f, 120.0, 120.0)) > 9.21


def coast(f: TrackFilter, seconds: float, step: float = 1.0 / 30.0) -> float:
    """Predict forward with no detections, as during a detector dropout."""
    t = f.t
    end = f.t + seconds
    while t < end:
        t = min(t + step, end)
        f.predict_to(t)
    return t


def test_gate_widens_while_coasting():
    f = run(simulate([1.0 / 30.0] * 30))
    before = f.center_mahalanobis2(offset_measurement(f, 0.0, 45.0))

    coast(f, 0.5)

    after = f.center_mahalanobis2(offset_measurement(f, 0.0, 45.0))
    assert after < before


def test_real_detection_after_dropout_passes_gate():
    """Half a second of misses, then the rocket reappears where physics put it."""
    f = run(simulate([1.0 / 30.0] * 30))
    t_reappear = coast(f, 0.5)

    cx, cy = truth_at(t_reappear)
    genuine = np.array([cx, cy, SIZE[0], SIZE[1]], dtype=np.float64)
    assert f.center_mahalanobis2(genuine) < 9.21


def test_gate_still_rejects_far_outlier_after_dropout():
    """Widening must not make the gate meaningless: smoke elsewhere stays out."""
    f = run(simulate([1.0 / 30.0] * 30))
    t_reappear = coast(f, 0.5)

    cx, cy = truth_at(t_reappear)
    smoke = np.array([cx + 300.0, cy + 300.0, SIZE[0], SIZE[1]], dtype=np.float64)
    assert f.center_mahalanobis2(smoke) > 9.21


def test_update_with_non_finite_measurement_flags_divergence():
    f = run(simulate([1.0 / 30.0] * 10))
    before = f.center
    assert f.diverged is False

    f.update(np.array([np.nan, np.nan, 40.0, 90.0]))

    assert f.diverged is True
    assert f.center == before
    assert np.all(np.isfinite(f.x))
