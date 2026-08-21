# Legacy scripts

One-off scripts from the pre-package layout. Prefer:

- `python -m scripts.run_track`
- `python -m scripts.run_bench`
- `python -m scripts.export_onnx`
- `python -m scripts.train`

| File | Notes |
|------|-------|
| `train.py` | Original 200-epoch trainer (logic preserved in `scripts/train.py`) |
| `test_tracking.py` | Ultralytics `model.track()` demo — not the default SORT path |
| `test.py` | Batch detect over `testing_media/` |
| `visualize.py` | Draw dataset labels |
| `bytetrack.yaml` | Ultralytics ByteTrack config for optional comparison |
