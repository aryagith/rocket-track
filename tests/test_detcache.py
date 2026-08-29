"""Detection cache: replay a detector pass without a GPU."""

from __future__ import annotations

import numpy as np
import pytest

from rocket_track.detcache import (
    CacheMismatch,
    CacheMeta,
    FrameDetections,
    read_cache,
    write_cache,
)


def make_meta(**kw) -> CacheMeta:
    fields = dict(
        source="testing_media/rocket_launch.mov",
        weights="weights/best.pt",
        imgsz=512,
        conf=0.30,
        fps=30.0,
        n_frames=3,
        frame_w=1920,
        frame_h=1080,
    )
    fields.update(kw)
    return CacheMeta(**fields)


def make_records():
    return [
        FrameDetections(0, 0.0, np.array([[10.0, 20.0, 50.0, 110.0, 0.91, 0.0]])),
        FrameDetections(1, 1 / 30, np.empty((0, 6))),
        FrameDetections(
            2,
            2 / 30,
            np.array([[12.0, 18.0, 52.0, 108.0, 0.88, 0.0], [1.0, 2.0, 3.0, 4.0, 0.4, 0.0]]),
        ),
    ]


def test_round_trips_records(tmp_path):
    path = tmp_path / "cache.jsonl"
    write_cache(path, make_meta(), make_records())

    meta, records = read_cache(path)

    assert meta == make_meta()
    assert [r.index for r in records] == [0, 1, 2]
    assert records[1].dets.shape == (0, 6)
    assert np.allclose(records[2].dets, make_records()[2].dets)
    assert records[2].t == pytest.approx(2 / 30)


def test_rejects_cache_built_for_different_settings(tmp_path):
    path = tmp_path / "cache.jsonl"
    write_cache(path, make_meta(), make_records())

    with pytest.raises(CacheMismatch):
        read_cache(path, expect=make_meta(imgsz=640))


def test_accepts_matching_expectation_across_machines(tmp_path):
    """Paths differ between machines; the clip name is what must match."""
    path = tmp_path / "cache.jsonl"
    write_cache(path, make_meta(source=r"C:\somewhere\rocket_launch.mov"), make_records())

    meta, _ = read_cache(path, expect=make_meta(source="/home/other/rocket_launch.mov"))

    assert meta.imgsz == 512


def test_missing_cache_reports_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_cache(tmp_path / "absent.jsonl")
