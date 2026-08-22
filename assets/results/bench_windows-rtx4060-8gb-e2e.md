# Benchmark results (windows-rtx4060-8gb-e2e)

| Backend | N | Warmup | Mean (ms) | Median (ms) | P95 (ms) | FPS | Peak VRAM (MB) | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| pytorch_cuda | 60 | 15 | 18.58 | 17.09 | 24.88 | 53.83 | 78.4 |  |
| pytorch_cpu | 60 | 15 | 99.29 | 96.52 | 115.67 | 10.07 | N/A |  |
