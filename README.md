# rocket-track

Detect and track rockets in launch video: domain-tuned YOLOv8, in-repo SORT, and honest latency benches.

**FPS goal:** ~60 detect+track on CUDA (30 is the real-time floor). On an RTX 4060 Laptop, **TensorRT clears that** (~90–98 e2e / ~105–118 infer).

![Tracked rocket](assets/results/demo_track_still.jpg)

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python -m scripts.run_track --source testing_media/rocket_launch.mov --device auto --profile fast --out outputs/tracked
```

Open `outputs/tracked/rocket_launch_tracked.mp4`.

| Profile | imgsz | Weights default | Use |
|---------|------:|-----------------|-----|
| `fast` (default) | 512 | `best.pt` (s) | Best speed/quality balance |
| `quality` | 640 | `best.pt` (s) | Better recall on tiny rockets |
| `realtime` | 416 | `best_n.pt` if present | Smaller input / nano |

`--device auto` picks CUDA when available. FP16 is on for GPU (`--half` / `--no-half`).

### Measured — RTX 4060 Laptop (`rocket_launch.mov`, 814 frames)

| Weights | Mode | e2e FPS | Infer FPS | Frames w/ tracks |
|---------|------|--------:|----------:|-----------------:|
| `best.engine` (s, TRT FP16) | detect+SORT (`--no-save`) | **90.0** | **105.3** | 229 (28.1%) |
| `best_n.engine` (n, TRT FP16) | detect+SORT (`--no-save`) | **98.4** | **118.0** | 168 (20.6%) |
| `best.pt` (s, FP16) | detect+SORT (`--no-save`) | 46.7 | 52.4 | 239 (29.4%) |
| `best_n.pt` (n, FP16) | detect+SORT (`--no-save`) | 44.6 | 49.0 | see note\* |
| `best.engine` (s) | detect+SORT+write MP4 | **72.3** | 110.4 | 229 (28.1%) |

\*PyTorch nano bake-off (`compare_speed`, `min_hits=1`) hit 290 frames; `run_track` uses `min_hits=3`. Nano is only ~1–3 FPS faster than s in PyTorch and loses recall — keep `fast` on `best.pt`. Engines are GPU-local (gitignored).

```bash
# PyTorch speed check
python -m scripts.run_track --source testing_media/rocket_launch.mov --device cuda --profile fast --half --no-save

# TensorRT (pip install tensorrt==10.3.0 && python -m scripts.export_engine)
python -m scripts.run_track --weights weights/best.engine --device cuda --imgsz 512 --half --out outputs/tracked

python -m pytest tests -q
```

## Layout

| Path | Purpose |
|------|---------|
| `src/rocket_track/` | Library: detect, SORT, pipeline, backends, bench |
| `scripts/` | CLIs (see below) |
| `weights/` | `best.pt` / `best_n.pt` (+ ONNX); `*.engine` local only |
| `testing_media/` | Launch clips (`rocket_launch.mov`, `testvid.mp4`) |
| `outputs/tracked/` | Annotated videos (gitignored) |
| `assets/` | Smoke still + bench CSV/plots |
| `configs/` | `default.yaml`, `fast.yaml`, `bytetrack.yaml` |
| `data.yaml` + `train/` `valid/` `test/` | Dataset (labels in git; images gitignored) |
| `runs/train/` | Training curves (`rocket_detector`, `rocket_detector_n`) |
| `docs/DATASET.md` | How to download full images |
| `tests/` | Unit tests |

## Scripts

| Command | Role |
|---------|------|
| `python -m scripts.run_track` | Detect + SORT (or ByteTrack) on video/image |
| `python -m scripts.run_bench` | Backend latency table |
| `python -m scripts.compare_speed` | Infer FPS bake-off across weight files |
| `python -m scripts.export_onnx` | Export `.onnx` |
| `python -m scripts.export_engine` | Export TensorRT `.engine` (GPU-local) |
| `python -m scripts.train` | Fine-tune YOLOv8s |
| `python -m scripts.train_nano` | Fine-tune YOLOv8n → `weights/best_n.pt` |
| `python -m scripts.smoke_check` | Quick detect/track sanity check |

## Weights

- `weights/best.pt` — default (YOLOv8s, class `Rocket`)
- `weights/best_n.pt` — YOLOv8n (40 epochs, imgsz 512); used by `realtime` when present
- `weights/best.onnx` / `best_n.onnx` — ONNX for benches
- `weights/*.engine` — TensorRT; rebuild per GPU

```bash
python -m scripts.export_onnx --weights weights/best.pt --out weights/best.onnx
python -m scripts.train_nano --data data.yaml
pip install tensorrt==10.3.0
python -m scripts.export_engine --weights weights/best.pt --imgsz 512
```

## Dataset

Single class `Rocket`. Labels are tracked; full images are not (~70GB). Details: [`docs/DATASET.md`](docs/DATASET.md).

Roboflow: `arbalesttest` / `rocket-tracking-pduic-ay8b4` v1 (CC BY 4.0).

| Model | mAP50 | mAP50-95 | Precision | Recall |
|-------|------:|---------:|----------:|-------:|
| YOLOv8s (epoch 200) | 0.882 | 0.661 | 0.928 | 0.790 |
| YOLOv8n (best epoch 39) | 0.838 | 0.612 | 0.878 | 0.760 |

## Tracking

Default tracker is **SORT** in `src/rocket_track/track_sort.py` (Kalman + IoU + Hungarian). After a track is confirmed, missed frames still draw a **coasted** Kalman prediction for `--coast-frames` (default 15; `0` = classic SORT). Coasted boxes are thinner and labeled with `~`. Optional: `--tracker bytetrack`.

## Benchmarks

```bash
python -m scripts.run_bench --source testing_media/testvid.mp4 --out assets/results/
python -m scripts.compare_speed --weights weights/best.pt weights/best_n.pt --device cuda
```

Detect-only (`imgsz=640`, earlier bench):

| Backend | FPS | Notes |
|---------|----:|-------|
| pytorch_cuda | 39.9 | |
| pytorch_cpu | 9.0 | |
| onnx_cpu | 6.9 | |
| onnx_cuda | — | EP not wired here |
| tensorrt (track e2e) | **105+** | See table above |

Artifacts: `assets/results/bench_windows-rtx4060-8gb-amd64.*`, `speed_compare.csv`.

Warmup frames are discarded. Don’t invent TensorRT numbers from another GPU.

## Limitations

- SORT has no ReID; IDs can switch under occlusion (coasting fills short detect gaps only)
- Full train images aren’t in git
- TensorRT engines are GPU-specific (`tensorrt==10.3.0` on this CUDA 12 stack; TRT 11 breaks Ultralytics export)

## License

MIT (`LICENSE`). Dataset CC BY 4.0. Acknowledgments: Ultralytics, Roboflow, SORT, Arbalest — see `NOTICE.md`.
