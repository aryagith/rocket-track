# Benchmarks

Machine-tagged artifacts live under `assets/results/`:

- `bench_<platform_id>.csv`
- `bench_<platform_id>.md`
- `bench_<platform_id>.png`

Generate on your machine:

```bash
python -m scripts.run_bench --source testing_media/testvid.mp4 --weights weights/best.pt --out assets/results/
```

## Protocol

- Same source, `imgsz`, `conf`, `iou` across backends
- Warmup frames discarded (default 30)
- Report mean / median / p95 latency (ms), FPS, peak CUDA VRAM when available
- Unavailable backends recorded as **N/A** with a reason (do not invent numbers)

## Thermal / fairness caveats

- Prefer a consistent power mode (e.g. Windows Best Performance) and a cool GPU before timing
- Laptop GPUs throttle under heat; discard the first timed batch if clocks are still ramping
- Do not mix cloud GPU numbers with laptop RTX 4060 numbers in one table
- ONNX CUDA EP and TensorRT require extra installs; absence is expected on many clones

Author checkout artifact: `assets/results/bench_windows-rtx4060-8gb-amd64.*` (RTX 4060 Laptop 8GB). Re-run on your machine before comparing.
