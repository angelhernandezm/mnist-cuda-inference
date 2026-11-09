# Benchmark: cpu (Intel64 Family 6 Model 170 Stepping 4, GenuineIntel, 16 threads)

torch 2.11.0+cu128

| Variant | Accuracy | batch 1: p50 ms / p99 ms / img/s | batch 32: p50 ms / p99 ms / img/s | batch 256: p50 ms / p99 ms / img/s |
|---|---|---|---|---|
| eager fp32 | 0.9834 | 0.655 / 3.681 / 1,117 | 7.222 / 51.005 / 2,412 | 38.645 / 50.379 / 6,442 |
| TorchScript (frozen) | 0.9834 | 0.441 / 2.218 / 1,705 | 13.020 / 54.555 / 1,678 | 47.341 / 56.853 / 5,575 |
| dynamic int8 (Linear layers) | 0.9834 | 1.370 / 5.963 / 582 | 6.428 / 13.323 / 4,640 | 42.728 / 98.207 / 4,789 |
| ONNX Runtime | 0.9834 | 0.130 / 0.605 / 6,501 | 1.002 / 18.486 / 15,093 | 7.839 / 28.657 / 26,324 |
