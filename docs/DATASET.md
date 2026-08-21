# Dataset

## What’s in git

| Asset | Tracked? | Notes |
|-------|----------|-------|
| `data.yaml` | yes | class `Rocket`, Roboflow metadata |
| `train/labels`, `valid/labels`, `test/labels` | yes | YOLO txt labels |
| `train/images`, `valid/images`, `test/images` | **no** | gitignored; too large (~70GB locally) |
| `assets/sample/` | yes | tiny smoke subset |
| `testing_media/` | yes | short clips / stills for track + bench |

## Obtain full images

From `data.yaml`:

- Workspace: `arbalesttest`
- Project: `rocket-tracking-pduic-ay8b4`
- Version: `1`
- Format: **YOLOv8**
- License: CC BY 4.0
- URL: https://app.roboflow.com/arbalesttest/rocket-tracking-pduic-ay8b4/1

Download/export, then extract so files land at:

```
train/images/*.jpg
valid/images/*.jpg
test/images/*.jpg
```

Keep existing label folders. If Roboflow renames files, re-export with the same version used for training or regenerate labels.

Approximate counts from the author’s checkout: train 2959 / valid 955 / test 365 images.

## Train without re-downloading

You do **not** need the full image dump to:

- run `python -m scripts.run_track`
- run `python -m scripts.run_bench`
- run unit tests

You **do** need images (or a private mirror) for `python -m scripts.train`.
