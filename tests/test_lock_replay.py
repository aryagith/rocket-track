"""Replay real launch detections through the lock tracker (no GPU needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rocket_track.detcache import read_cache
from rocket_track.lock_track import LockPhase, LockTracker

CACHE = Path(__file__).resolve().parents[1] / "assets" / "caches" / "rocket_launch_best_512_c30.jsonl"
COAST_S = 0.25  # 15 frames at the clip's ~59 FPS, matching the SORT default


@pytest.fixture(scope="module")
def cache():
    if not CACHE.exists():
        pytest.skip(f"detection cache missing: {CACHE} (build with scripts.cache_dets)")
    return read_cache(CACHE)


def run_tracker(cache, **kw):
    meta, records = cache
    tracker = LockTracker(frame_size=meta.frame_size, coast_s=COAST_S, **kw)
    return [(r, tracker.update(r.dets, r.t)) for r in records]


def test_locks_onto_the_launch(cache):
    results = run_tracker(cache)
    assert any(s.phase is LockPhase.LOCKED for _, s in results)


def test_never_emits_beyond_the_coast_window(cache):
    """A published box is always backed by a detection inside the coast window."""
    results = run_tracker(cache)

    last_det_t = None
    for record, state in results:
        if record.dets.shape[0] > 0:
            last_det_t = record.t
        if state.has_target:
            assert last_det_t is not None
            assert record.t - last_det_t <= COAST_S + 1e-9


def test_estimates_stay_physically_plausible(cache):
    """Coasting may leave frame, but a diverging filter would produce nonsense."""
    meta, _ = cache
    frame_w, frame_h = meta.frame_size

    for _, state in run_tracker(cache):
        if not state.has_target:
            continue
        box = np.array(state.xyxy, dtype=np.float64)
        assert np.all(np.isfinite(box))
        assert 0.0 < box[2] - box[0] <= frame_w
        assert 0.0 < box[3] - box[1] <= frame_h
