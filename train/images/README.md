# Dataset images (not in git)

Full Roboflow image dumps are **gitignored** (~tens of GB).

1. Export YOLOv8 format from Roboflow project `arbalesttest/rocket-tracking-pduic-ay8b4` (v1).
2. Unpack so images land here as `*.jpg` / `*.png` beside the existing `../labels/` files.
3. Confirm `data.yaml` still points at `train/images`, `valid/images`, `test/images`.

Smoke runs do **not** need this folder — use `assets/sample/` or `testing_media/`.
See [`docs/DATASET.md`](../../docs/DATASET.md).
