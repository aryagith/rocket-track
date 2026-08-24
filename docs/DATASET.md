# Dataset

## In git

| Asset | Tracked? |
|-------|----------|
| `data.yaml` | yes |
| `train/labels`, `valid/labels`, `test/labels` | yes |
| `train/images`, `valid/images`, `test/images` | **no** (~70GB) |
| `assets/sample/demo_rocket.jpg` | yes |
| `testing_media/rocket_launch.mov`, `testvid.mp4` | yes |

## Download images

Roboflow (`data.yaml`): workspace `arbalesttest`, project `rocket-tracking-pduic-ay8b4`, version `1`, YOLOv8 export, CC BY 4.0.

Unpack into:

```
train/images/
valid/images/
test/images/
```

Approx. counts: train 2959 / valid 955 / test 365.

Track / bench / tests work without the full image dump. Retrain needs images (or a private mirror).

## Hard frames for the next train

After a video run, dump detector misses (local only, under `outputs/`):

```bash
python -m scripts.export_hard_frames --source testing_media/rocket_launch.mov --device cuda --imgsz 640 --conf 0.2
```

Label those JPEGs, merge into `train/images` + `train/labels`, then `python -m scripts.train` / `train_nano`.
