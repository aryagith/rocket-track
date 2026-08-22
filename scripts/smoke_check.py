"""Quick clone smoke check: paths, imports, optional one-frame track."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts._paths import ROOT, default_still, default_weights, resolve_device


def main() -> int:
    p = argparse.ArgumentParser(description="rocket-track smoke check")
    p.add_argument("--track", action="store_true", help="Run one-frame SORT track if weights exist")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    ok = True
    checks = {
        "data.yaml": ROOT / "data.yaml",
        "sample assets": ROOT / "assets" / "sample",
        "weights/best.pt": ROOT / "weights" / "best.pt",
        "src package": ROOT / "src" / "rocket_track" / "track_sort.py",
    }
    for name, path in checks.items():
        exists = path.exists()
        print(f"[{'OK' if exists else 'MISS'}] {name}: {path.relative_to(ROOT)}")
        if name != "weights/best.pt" and not exists:
            ok = False

    try:
        from rocket_track.track_sort import SortTracker  # noqa: F401

        print("[OK] import rocket_track.track_sort")
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] import SORT: {exc}")
        ok = False

    weights = default_weights()
    if args.track:
        if not weights.exists():
            print(f"[SKIP] --track requested but weights missing: {weights}")
        else:
            from rocket_track.pipeline import TrackPipeline

            source = default_still()
            out = ROOT / "outputs" / "smoke"
            device = resolve_device(args.device)
            print(f"Tracking {source} on device={device} ...")
            path = TrackPipeline(weights=weights, device=device, tracker="sort", min_hits=1).run(
                source, out
            )
            print(f"[OK] wrote {path}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
