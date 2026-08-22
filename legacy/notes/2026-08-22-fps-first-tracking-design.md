# FPS-first tracking (software)

## Target

- **≥60 FPS** end-to-end detect+SORT on RTX 4060 CUDA for launch video (comfortable headroom for a future pan-tilt loop).
- **≥30 FPS** is the hard floor for “real-time”; below that, shrink model/`imgsz` before buying Jetson hopes.
- Jetson Orin Nano is for watts/TensorRT, not a free FPS upgrade over a 4060 laptop.

## Levers (this repo)

1. `imgsz` 640 → 512 (fast profile) while watching small-rocket recall
2. CUDA FP16 (`half=True`) on the detector
3. Keep SORT in-process (cheap vs YOLO)
4. Write annotated video to `outputs/tracked/` at source FPS
5. Report mean end-to-end FPS after each `run_track`

Primary demo source: `testing_media/rocket launch.mov` (fallback: `testvid.mp4`).
