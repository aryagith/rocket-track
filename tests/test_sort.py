"""Unit tests for in-repo SORT tracker."""

from __future__ import annotations

import numpy as np

from rocket_track.track_sort import SortTracker, iou_batch


def test_iou_perfect_overlap():
    a = np.array([[0, 0, 10, 10]], dtype=np.float64)
    b = np.array([[0, 0, 10, 10]], dtype=np.float64)
    assert iou_batch(a, b)[0, 0] == 1.0


def test_iou_no_overlap():
    a = np.array([[0, 0, 10, 10]], dtype=np.float64)
    b = np.array([[20, 20, 30, 30]], dtype=np.float64)
    assert iou_batch(a, b)[0, 0] == 0.0


def test_sort_stable_id_on_moving_box():
    tracker = SortTracker(max_age=5, min_hits=1, iou_threshold=0.3)
    ids = []
    for t in range(10):
        x1, y1 = 100 + 5 * t, 100 + 2 * t
        dets = np.array([[x1, y1, x1 + 40, y1 + 80, 0.9, 0]], dtype=np.float64)
        tracks = tracker.update(dets)
        assert len(tracks) == 1
        ids.append(tracks[0].track_id)
    assert len(set(ids)) == 1


def test_sort_new_id_for_distant_box():
    tracker = SortTracker(max_age=5, min_hits=1, iou_threshold=0.3)
    a = tracker.update(np.array([[10, 10, 50, 90, 0.9, 0]], dtype=np.float64))
    b = tracker.update(np.array([[400, 300, 440, 380, 0.9, 0]], dtype=np.float64))
    assert a[0].track_id != b[0].track_id


def test_sort_empty_detections_keeps_coasting():
    tracker = SortTracker(max_age=5, min_hits=1, iou_threshold=0.3, coast_frames=3)
    tracker.update(np.array([[10, 10, 50, 90, 0.9, 0]], dtype=np.float64))
    # Confirmed (min_hits=1): empty frames still emit predicted boxes for coast_frames
    for _ in range(3):
        tracks = tracker.update(np.empty((0, 6)))
        assert len(tracks) == 1
        assert tracks[0].coasted is True
    # Past coast_frames but within max_age: alive, not drawn
    assert tracker.update(np.empty((0, 6))) == []
    # Eventually expires past max_age
    for _ in range(5):
        tracker.update(np.empty((0, 6)))
    assert tracker.update(np.empty((0, 6))) == []


def test_sort_coast_requires_min_hits():
    tracker = SortTracker(max_age=10, min_hits=3, iou_threshold=0.3, coast_frames=5)
    # Only one hit — not confirmed; miss should not coast
    tracker.update(np.array([[10, 10, 50, 90, 0.9, 0]], dtype=np.float64))
    assert tracker.update(np.empty((0, 6))) == []
    # Confirm with three matches, then coast
    tr = SortTracker(max_age=10, min_hits=3, iou_threshold=0.3, coast_frames=5)
    for t in range(3):
        x = 10 + 2 * t
        out = tr.update(np.array([[x, 10, x + 40, 90, 0.9, 0]], dtype=np.float64))
        assert len(out) == 1
        assert out[0].coasted is False
    coasted = tr.update(np.empty((0, 6)))
    assert len(coasted) == 1 and coasted[0].coasted is True
