"""Benchmark inference backends on rocket sample media."""

from __future__ import annotations

import csv
import platform
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .backends import (
    create_backend,
    list_backend_availability,
    peak_vram_mb,
    reset_vram_peak,
)
from .metrics import LatencyStats, stats_to_markdown, summarize_latencies
from .pipeline import IMAGE_EXTS, VIDEO_EXTS, is_image, is_video

PathLike = Union[str, Path]


def default_platform_id() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    gpu = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0).lower().replace(" ", "")
            # e.g. nvidiageforcertx4060laptopgpu -> rtx4060
            if "4060" in name:
                gpu = "rtx4060-8gb"
            else:
                gpu = name[:24]
    except Exception:  # noqa: BLE001
        pass
    return f"{system}-{gpu}-{machine}"


def load_bench_frames(source: PathLike, max_frames: int = 120) -> List[np.ndarray]:
    source = Path(source)
    frames: List[np.ndarray] = []
    if source.is_dir():
        files = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        for p in files[:max_frames]:
            img = cv2.imread(str(p))
            if img is not None:
                frames.append(img)
        return frames
    if is_image(source):
        img = cv2.imread(str(source))
        if img is None:
            raise FileNotFoundError(source)
        return [img]
    if is_video(source) or source.suffix.lower() in VIDEO_EXTS:
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise FileNotFoundError(source)
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        return frames
    raise FileNotFoundError(source)


def run_backend_bench(
    backend_name: str,
    weights: PathLike,
    frames: Sequence[np.ndarray],
    warmup: int = 30,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.45,
    onnx_path: Optional[PathLike] = None,
) -> LatencyStats:
    avail = {i.name: i for i in list_backend_availability()}
    info = avail.get(backend_name)
    if backend_name == "onnx":
        backend_name = "onnx_cpu"
        info = avail.get("onnx_cpu")
    if backend_name == "tensorrt":
        return LatencyStats(
            backend="tensorrt",
            n_frames=0,
            warmup=warmup,
            mean_ms=float("nan"),
            median_ms=float("nan"),
            p95_ms=float("nan"),
            fps=float("nan"),
            peak_vram_mb=None,
            notes="N/A — TensorRT engine not packaged; skip unless you export format=engine locally",
        )
    if info is not None and not info.available:
        return LatencyStats(
            backend=backend_name,
            n_frames=0,
            warmup=warmup,
            mean_ms=float("nan"),
            median_ms=float("nan"),
            p95_ms=float("nan"),
            fps=float("nan"),
            peak_vram_mb=None,
            notes=f"N/A — {info.reason}",
        )

    try:
        backend = create_backend(
            backend_name, weights, imgsz=imgsz, conf=conf, iou=iou, onnx_path=onnx_path
        )
    except Exception as exc:  # noqa: BLE001
        return LatencyStats(
            backend=backend_name,
            n_frames=0,
            warmup=warmup,
            mean_ms=float("nan"),
            median_ms=float("nan"),
            p95_ms=float("nan"),
            fps=float("nan"),
            peak_vram_mb=None,
            notes=f"N/A — {exc}",
        )

    reset_vram_peak()
    # Warmup (discarded)
    for i in range(min(warmup, len(frames))):
        backend.infer(frames[i % len(frames)])

    times: List[float] = []
    # Timed loop — cycle frames if fewer than desired
    n_timed = max(len(frames), 1)
    for i in range(n_timed):
        _, dt = backend.timed_infer(frames[i % len(frames)])
        times.append(dt)

    vram = peak_vram_mb() if "cuda" in backend_name else None
    return summarize_latencies(backend.name, times, warmup=warmup, peak_vram_mb=vram)


def write_results(
    rows: List[LatencyStats],
    out_dir: PathLike,
    platform_id: Optional[str] = None,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    platform_id = platform_id or default_platform_id()
    csv_path = out_dir / f"bench_{platform_id}.csv"
    md_path = out_dir / f"bench_{platform_id}.md"
    png_path = out_dir / f"bench_{platform_id}.png"

    fieldnames = [
        "backend",
        "n_frames",
        "warmup",
        "mean_ms",
        "median_ms",
        "p95_ms",
        "fps",
        "peak_vram_mb",
        "notes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r.to_dict())

    md_path.write_text(
        stats_to_markdown(rows, title=f"Benchmark results ({platform_id})"),
        encoding="utf-8",
    )

    # Chart only backends with numeric means
    labels, means = [], []
    for r in rows:
        if r.n_frames > 0 and np.isfinite(r.mean_ms):
            labels.append(r.backend)
            means.append(r.mean_ms)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if labels:
        ax.bar(labels, means, color="#2c5f7c")
        ax.set_ylabel("Mean latency (ms)")
        ax.set_title(f"Rocket detector latency — {platform_id}")
        ax.tick_params(axis="x", rotation=20)
    else:
        ax.text(0.5, 0.5, "No successful backends", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(png_path, dpi=140)
    plt.close(fig)

    return {"csv": csv_path, "md": md_path, "png": png_path, "platform_id": platform_id}


def run_bench(
    source: PathLike,
    weights: PathLike,
    backends: Iterable[str],
    out_dir: PathLike,
    warmup: int = 30,
    max_frames: int = 120,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.45,
    platform_id: Optional[str] = None,
    onnx_path: Optional[PathLike] = None,
) -> dict:
    frames = load_bench_frames(source, max_frames=max_frames)
    if not frames:
        raise RuntimeError(f"No frames loaded from {source}")
    rows = [
        run_backend_bench(
            b.strip(),
            weights,
            frames,
            warmup=warmup,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            onnx_path=onnx_path,
        )
        for b in backends
    ]
    paths = write_results(rows, out_dir, platform_id=platform_id)
    paths["rows"] = rows
    return paths
