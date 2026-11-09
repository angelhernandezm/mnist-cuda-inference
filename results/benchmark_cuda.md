# Benchmark: cuda (NVIDIA RTX A500 Laptop GPU)

torch 2.11.0+cu128

| Variant | Accuracy | batch 1: p50 ms / p99 ms / img/s | batch 32: p50 ms / p99 ms / img/s | batch 256: p50 ms / p99 ms / img/s |
|---|---|---|---|---|
| CUDA eager fp32 | 0.9834 | 0.555 / 1.760 / 1,464 | 0.560 / 1.577 / 50,651 | 2.805 / 3.151 / 90,020 |
| CUDA eager fp16 | 0.9834 | 0.380 / 0.686 / 2,548 | 0.433 / 1.184 / 69,434 | 1.815 / 2.201 / 139,364 |
| CUDA Graphs fp32 | 0.9834 | 0.104 / 0.240 / 8,261 | 0.424 / 0.507 / 74,867 | 2.694 / 3.075 / 93,542 |
