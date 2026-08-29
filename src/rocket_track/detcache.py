"""Persist one detector pass over a clip so tuning never re-runs the network.

A cache is JSONL: a header record describing how the detections were produced,
then one record per frame. Replaying a cache turns a filter parameter sweep from
a GPU job into a CPU job that finishes in milliseconds, and lets tests exercise
the tracker against real launch detections.

Caches are only comparable when they came from the same clip and the same
detector settings, so reads can assert that up front rather than silently
scoring the wrong thing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, Path]

CACHE_KIND = "rocket-track-detcache"
CACHE_VERSION = 1
DET_COLS = 6  # x1, y1, x2, y2, score, class_id


class CacheMismatch(RuntimeError):
    """Raised when a cache was built for a different clip or detector setup."""


def _basename(path: str) -> str:
    """Last path component, tolerant of Windows and POSIX separators."""
    return re.split(r"[\\/]", str(path))[-1]


@dataclass(frozen=True)
class CacheMeta:
    source: str
    weights: str
    imgsz: int
    conf: float
    fps: float
    n_frames: int
    frame_w: int = 0
    frame_h: int = 0

    @property
    def frame_size(self) -> Tuple[int, int]:
        return int(self.frame_w), int(self.frame_h)

    @property
    def clip_name(self) -> str:
        return _basename(self.source)

    @property
    def weights_name(self) -> str:
        return _basename(self.weights)

    def matches(self, other: "CacheMeta") -> bool:
        """Compare on identity, not on absolute paths that differ per machine."""
        return (
            self.clip_name == other.clip_name
            and self.weights_name == other.weights_name
            and self.imgsz == other.imgsz
            and abs(self.conf - other.conf) < 1e-9
        )

    def to_dict(self) -> dict:
        return {
            "kind": CACHE_KIND,
            "version": CACHE_VERSION,
            "source": self.source,
            "weights": self.weights,
            "imgsz": self.imgsz,
            "conf": self.conf,
            "fps": self.fps,
            "n_frames": self.n_frames,
            "frame_w": self.frame_w,
            "frame_h": self.frame_h,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CacheMeta":
        return cls(
            source=str(d["source"]),
            weights=str(d["weights"]),
            imgsz=int(d["imgsz"]),
            conf=float(d["conf"]),
            fps=float(d["fps"]),
            n_frames=int(d["n_frames"]),
            frame_w=int(d.get("frame_w", 0)),
            frame_h=int(d.get("frame_h", 0)),
        )


@dataclass
class FrameDetections:
    index: int
    t: float
    dets: np.ndarray  # (N, 6)

    def __post_init__(self) -> None:
        arr = np.asarray(self.dets, dtype=np.float64)
        if arr.size == 0:
            arr = np.empty((0, DET_COLS), dtype=np.float64)
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)
        self.dets = arr


def write_cache(path: PathLike, meta: CacheMeta, records: Iterable[FrameDetections]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(meta.to_dict()) + "\n")
        for r in records:
            row = {"i": int(r.index), "t": float(r.t), "d": r.dets.tolist()}
            f.write(json.dumps(row) + "\n")
    return path


def read_cache(
    path: PathLike, expect: Optional[CacheMeta] = None
) -> Tuple[CacheMeta, List[FrameDetections]]:
    """Load a cache, optionally asserting it was built for ``expect``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Detection cache not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        lines = [ln for ln in (line.strip() for line in f) if ln]

    if not lines:
        raise CacheMismatch(f"Detection cache is empty: {path}")

    header = json.loads(lines[0])
    if header.get("kind") != CACHE_KIND:
        raise CacheMismatch(f"Not a rocket-track detection cache: {path}")
    if int(header.get("version", 0)) != CACHE_VERSION:
        raise CacheMismatch(
            f"Cache version {header.get('version')} != {CACHE_VERSION}; rebuild {path}"
        )

    meta = CacheMeta.from_dict(header)
    if expect is not None and not meta.matches(expect):
        raise CacheMismatch(
            f"Cache {path} was built for clip={meta.clip_name} weights={meta.weights_name} "
            f"imgsz={meta.imgsz} conf={meta.conf}, but clip={expect.clip_name} "
            f"weights={expect.weights_name} imgsz={expect.imgsz} conf={expect.conf} was requested. "
            "Rebuild it with: python -m scripts.cache_dets"
        )

    records = [
        FrameDetections(index=int(row["i"]), t=float(row["t"]), dets=np.asarray(row["d"]))
        for row in (json.loads(ln) for ln in lines[1:])
    ]
    return meta, records


def replay(
    records: Sequence[FrameDetections], tracker, reset: bool = True
) -> List:
    """Feed cached detections through anything with ``update(dets, t)``."""
    if reset and hasattr(tracker, "reset"):
        tracker.reset()
    return [tracker.update(r.dets, r.t) for r in records]
