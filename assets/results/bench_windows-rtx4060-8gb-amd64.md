# Benchmark results (windows-rtx4060-8gb-amd64)

| Backend | N | Warmup | Mean (ms) | Median (ms) | P95 (ms) | FPS | Peak VRAM (MB) | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| pytorch_cpu | 40 | 10 | 110.75 | 110.20 | 116.46 | 9.03 | N/A |  |
| pytorch_cuda | 40 | 10 | 25.08 | 24.82 | 31.71 | 39.88 | 78.4 |  |
| onnx_cpu | 40 | 10 | 144.33 | 139.14 | 185.45 | 6.93 | N/A |  |
| onnx_cuda | 0 | 10 | nan | nan | nan | nan | N/A | N/A — CUDA EP not installed |
| tensorrt | 0 | 10 | nan | nan | nan | nan | N/A | N/A — TensorRT engine not packaged; skip unless you export format=engine locally |
