# Legacy

Old one-off scripts and notes kept for reference. Prefer the root CLIs:

- `python -m scripts.run_track`
- `python -m scripts.run_bench`
- `python -m scripts.export_onnx`
- `python -m scripts.train`

| Path | What it is |
|------|------------|
| `train.py` | Original 200-epoch trainer (logic also in `scripts/train.py`) |
| `test_tracking.py` | Ultralytics `model.track()` demo |
| `test.py` / `visualize.py` | Early detect / label viz scripts |
| `bytetrack.yaml` | Copy of ByteTrack config (active copy: `configs/bytetrack.yaml`) |
| `README.roboflow.txt` | Original Roboflow export blurb |
| `docs/` | Older architecture / bench writeups |
| `notes/` | Design notes from the FPS-first pass |
