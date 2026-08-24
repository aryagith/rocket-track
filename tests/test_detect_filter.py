"""Tests for detection post-filters (smoke / oversized FP rejection)."""

from __future__ import annotations

from rocket_track.detect import Detection, filter_detections


def test_filter_rejects_huge_smoke_box():
    # 1280x720 frame; rocket is small, smoke spans most of the trail
    rocket = Detection(xyxy=(600, 40, 640, 120), score=0.55, class_id=0)
    smoke = Detection(xyxy=(100, 100, 700, 600), score=0.70, class_id=0)  # huge
    kept = filter_detections([smoke, rocket], (720, 1280, 3), max_area_frac=0.04, max_det=3)
    assert len(kept) == 1
    assert kept[0].score == 0.55


def test_filter_keeps_top_max_det():
    dets = [
        Detection(xyxy=(10, 10, 30, 50), score=0.9, class_id=0),
        Detection(xyxy=(40, 10, 60, 50), score=0.8, class_id=0),
        Detection(xyxy=(70, 10, 90, 50), score=0.7, class_id=0),
    ]
    kept = filter_detections(dets, (720, 1280, 3), max_area_frac=0.5, max_det=2)
    assert [d.score for d in kept] == [0.9, 0.8]
