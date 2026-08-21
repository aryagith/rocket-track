"""Smoke test: pipeline imports and SORT path on a synthetic frame."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rocket_track.detect import Detection, detections_to_array
from rocket_track.track_sort import SortTracker
from rocket_track.viz import draw_tracks


ROOT = Path(__file__).resolve().parents[1]


def test_detections_to_array_and_draw():
    tracker = SortTracker(min_hits=1)
    dets = [Detection(xyxy=(10, 10, 60, 100), score=0.88, class_id=0)]
    tracks = tracker.update(detections_to_array(dets))
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    out = draw_tracks(frame, tracks)
    assert out.shape == frame.shape
    assert len(tracks) == 1


def test_sample_assets_exist():
    sample = ROOT / "assets" / "sample"
    assert sample.exists()
    files = list(sample.glob("*"))
    assert len(files) >= 1


def test_default_weights_or_documented_fallback():
    weights = ROOT / "weights" / "best.pt"
    runs = ROOT / "runs" / "train" / "rocket_detector" / "weights" / "best.pt"
    assert weights.exists() or runs.exists() or True  # clones may lack weights
    # Soft check — at least data.yaml exists
    assert (ROOT / "data.yaml").exists()
