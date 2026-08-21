"""Latency / FPS summary helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class LatencyStats:
    backend: str
    n_frames: int
    warmup: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    fps: float
    peak_vram_mb: Optional[float]
    notes: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


def summarize_latencies(
    backend: str,
    times_ms: Sequence[float],
    warmup: int = 0,
    peak_vram_mb: Optional[float] = None,
    notes: str = "",
) -> LatencyStats:
    arr = np.asarray(list(times_ms), dtype=np.float64)
    if arr.size == 0:
        return LatencyStats(
            backend=backend,
            n_frames=0,
            warmup=warmup,
            mean_ms=float("nan"),
            median_ms=float("nan"),
            p95_ms=float("nan"),
            fps=float("nan"),
            peak_vram_mb=peak_vram_mb,
            notes=notes or "no timed frames",
        )
    mean = float(np.mean(arr))
    return LatencyStats(
        backend=backend,
        n_frames=int(arr.size),
        warmup=warmup,
        mean_ms=mean,
        median_ms=float(np.median(arr)),
        p95_ms=float(np.percentile(arr, 95)),
        fps=1000.0 / mean if mean > 0 else float("nan"),
        peak_vram_mb=peak_vram_mb,
        notes=notes,
    )


def stats_to_markdown(rows: List[LatencyStats], title: str = "Benchmark results") -> str:
    lines = [
        f"# {title}",
        "",
        "| Backend | N | Warmup | Mean (ms) | Median (ms) | P95 (ms) | FPS | Peak VRAM (MB) | Notes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        vram = f"{r.peak_vram_mb:.1f}" if r.peak_vram_mb is not None else "N/A"
        lines.append(
            f"| {r.backend} | {r.n_frames} | {r.warmup} | {r.mean_ms:.2f} | "
            f"{r.median_ms:.2f} | {r.p95_ms:.2f} | {r.fps:.2f} | {vram} | {r.notes} |"
        )
    lines.append("")
    return "\n".join(lines)
