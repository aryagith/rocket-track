# rocket-track

Detect and track rockets in launch video: domain-tuned YOLOv8s, an in-repo SORT tracker, and latency benches on an RTX 4060.

**FPS goal:** ~60 end-to-end on CUDA (detect + track). 30 is the real-time floor.

![Tracked rocket](assets/results/demo_track_still.jpg)

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python -m scripts.run_track --source testing_media/rocket_launch.mov --device auto --profile fast --out outputs/tracked
```

Open `outputs/tracked/rocket_launch_tracked.mp4`.

| Profile | imgsz | Use |
|---------|------:|-----|
| `fast` (default) | 512 | Speed |
| `quality` | 640 | Better recall on tiny rockets |
| `realtime` | 416 | Max FPS |

`--device auto` uses CUDA when PyTorch sees a GPU, otherwise CPU. FP16 is enabled on CUDA.

```bash
python -m pytest tests -q
```

## Layout

| Path | Purpose |
|------|---------|
| `src/rocket_track/` | Library: detect, SORT, pipeline, backends, bench |
| `scripts/` | CLIs (`run_track`, `run_bench`, `export_onnx`, `train`, `smoke_check`) |
| `weights/best.pt` / `best.onnx` | Product detector |
| `testing_media/` | Launch clips (`rocket_launch.mov`, `testvid.mp4`) |
| `outputs/tracked/` | Annotated videos (gitignored, local only) |
| `assets/sample/` | Smoke still (`demo_rocket.jpg`) |
| `assets/results/` | Bench CSV/plots + demo stills |
| `configs/` | `fast.yaml`, `default.yaml`, `bytetrack.yaml` |
| `data.yaml` + `train/` `valid/` `test/` | Dataset (labels in git; images gitignored) |
| `runs/train/rocket_detector/` | Training curves / metrics |
| `docs/DATASET.md` | How to download full images |
| `legacy/` | Old scripts and notes |
| `tests/` | Unit tests |

## Weights

- `weights/best.pt` — default (fine-tuned YOLOv8s, class `Rocket`)
- `weights/best.onnx` — for ONNX benches

```bash
python -m scripts.export_onnx --weights weights/best.pt --out weights/best.onnx
```

## Dataset

Single class `Rocket`. Labels are tracked; full images are not (~70GB). Details: [`docs/DATASET.md`](docs/DATASET.md).

Roboflow: `arbalesttest` / `rocket-tracking-pduic-ay8b4` v1 (CC BY 4.0).

Val (epoch 200, from `runs/train/rocket_detector/results.csv`):

| mAP50 | mAP50-95 | Precision | Recall |
|------:|---------:|----------:|-------:|
| 0.882 | 0.661 | 0.928 | 0.790 |

## Tracking

Default tracker is **SORT** in `src/rocket_track/track_sort.py` (Kalman + IoU + Hungarian). Optional: `--tracker bytetrack` (Ultralytics).

## Benchmarks

```bash
python -m scripts.run_bench --source testing_media/testvid.mp4 --out assets/results/
```

RTX 4060 laptop (`imgsz=640`, detect-only):

| Backend | FPS | Notes |
|---------|----:|-------|
| pytorch_cuda | 39.9 | |
| pytorch_cpu | 9.0 | |
| onnx_cpu | 6.9 | |
| onnx_cuda / tensorrt | — | N/A until EP / engine available |

Artifacts: `assets/results/bench_windows-rtx4060-8gb-amd64.*`.

Don’t mix machines or invent TensorRT numbers. Warmup frames are discarded.

## CLI

```bash
python -m scripts.run_track --source testing_media/rocket_launch.mov --profile fast --device auto --out outputs/tracked
python -m scripts.run_bench --source testing_media/rocket_launch.mov --out assets/results/
python -m scripts.export_onnx
python -m scripts.smoke_check --track
python -m scripts.train --data data.yaml
```

## Limitations

- SORT has no ReID; IDs can switch under occlusion
- Full train images aren’t in git
- Jetson / TensorRT numbers only when measured on device

## License

MIT (`LICENSE`). Dataset CC BY 4.0. Acknowledgments: Ultralytics, Roboflow, SORT, Arbalest — see `NOTICE.md`.
