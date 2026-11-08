# MNIST CNN: from PyTorch to hand-written CUDA

One small CNN taken through the whole path a production model takes:

1. **Build** it in PyTorch: train, validate, test (98.7% on the 10k MNIST test set).
2. **Inference engineering**: export it (ONNX, TorchScript, a custom binary format), serve
   it several ways and measure latency, throughput and accuracy for each.
3. **CUDA and C++**: a standalone engine with four hand-written CUDA kernels (including a
   fused conv + ReLU + max-pool kernel) that loads the exported artifact and checks every
   prediction against PyTorch.

```
python/train.py ──► models/mnist_cnn.pt
python/export.py ──► artifacts/mnist_cnn.bin     ──► cpp/ mnist_infer (CPU or CUDA engine)
                     artifacts/mnist_test.bin        (test images + PyTorch's logits, for verification)
                     artifacts/mnist_cnn.onnx    ──► ONNX Runtime (and TensorRT, later)
                     artifacts/mnist_cnn_ts.pt   ──► TorchScript / libtorch
python/benchmark.py ──► results/benchmark_<device>.md
```

## The model

The network from the architecture diagram, 300,426 parameters:

| Layer | Output | Parameters | Multiply-adds per image |
|---|---|---|---|
| conv1 3×3, 1→16, ReLU | 16×26×26 | 160 | 97 k |
| conv2 3×3, 16→32, ReLU | 32×24×24 | 4,640 | **2.65 M (87%)** |
| max pool 2×2, dropout, flatten | 4608 | 0 | – |
| fc1 4608→64, ReLU, dropout | 64 | 294,976 (98%) | 295 k |
| fc2 64→10 | 10 | 650 | 0.6 k |

That table shows one of the main ideas in inference engineering: **parameters and compute are
in different places**. fc1 holds 98% of the weights but conv2 does 87% of the work, so conv2
is the kernel worth optimising, and fc1 is where weight compression (int8) saves memory.

The model returns raw logits, unlike the diagram's `log_softmax`: `CrossEntropyLoss`
applies log-softmax itself, and the predicted class (argmax) is the same either way.

## Quick start on Windows (Visual Studio + CUDA Toolkit)

Requirements: Python 3.10+, Visual Studio 2022 with "Desktop development with C++",
the CUDA Toolkit, CMake 3.24+ (bundled with VS 2022), and an NVIDIA GPU.

### 1. Python: train, export, benchmark

```powershell
cd C:\Code\angelhernandezm\mnist-cuda-inference
python -m venv .venv
.venv\Scripts\Activate.ps1

# PyTorch with CUDA: copy the exact command for your CUDA version from
# https://pytorch.org/get-started/locally/ , e.g.:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install onnx onnxscript onnxruntime

cd python
python train.py --epochs 3           # downloads MNIST, uses the GPU if torch sees one
python export.py                     # writes ..\artifacts\
python benchmark.py                  # CPU variants  -> ..\results\benchmark_cpu.md
python benchmark.py --device cuda    # GPU variants  -> ..\results\benchmark_cuda.md
```

### 2. C++/CUDA: build and run the engine

Open **"x64 Native Tools Command Prompt for VS 2022"** (so `cl.exe` and `nvcc` are on PATH):

```bat
cd C:\Code\angelhernandezm\mnist-cuda-inference
cmake -S cpp -B cpp\build -G "Visual Studio 17 2022" -A x64
cmake --build cpp\build --config Release
cpp\build\Release\mnist_infer.exe --artifacts artifacts
```

CMake should print `CUDA engine: enabled (architectures: native)`. If it says disabled,
it didn't find `nvcc`: check that the CUDA Toolkit's `bin` folder is on PATH. `native`
builds for the GPU in your laptop; to build for another GPU, add for example
`-DCMAKE_CUDA_ARCHITECTURES=86`.

Every run ends with `verification PASS` or `FAIL` for each engine: PASS means all
10,000 predictions match PyTorch and no logit differs by more than 0.001.

### On Linux / Raspberry Pi (CPU engine only)

```bash
cmake -S cpp -B cpp/build && cmake --build cpp/build -j
./cpp/build/mnist_infer --artifacts artifacts --engine cpu
```

## Results so far (CPU, 4-thread cloud VM; no GPU there)

Python serving options (`results/benchmark_cpu.md`):

| Variant | Accuracy | batch 1: p50 ms | batch 256: images/s |
|---|---|---|---|
| eager PyTorch fp32 | 0.9866 | 0.139 | 23,860 |
| TorchScript (frozen) | 0.9866 | 0.130 | 26,918 |
| dynamic int8 (Linear layers) | 0.9867 | 0.211 | 25,560 |
| **ONNX Runtime** | 0.9866 | **0.057** | **75,219** |

C++ CPU reference engine (`results/cpp_cpu.txt`): 10,472 images/s, all 10,000 predictions
identical to PyTorch, max logit difference 1.3e-5.

What these numbers say:

- **ONNX Runtime is 2.4–3× faster than PyTorch here**, from the same weights. It fuses
  operators and picks kernels ahead of time, while eager PyTorch pays framework overhead
  on every call. That overhead dominates at batch 1, where the model itself is tiny.
- **int8 made it slower at batch 1.** Dynamic quantization only covers the Linear layers,
  but 87% of the compute is in conv2, and quantizing activations at run time costs more
  than it saves on a model this small. It pays off for large, Linear-heavy models
  (transformers): measure before assuming.
- **Batching matters more than anything else**: batch 256 gives about 4× the images/s of
  batch 1 for every variant.
- **The naive C++ engine is slower than PyTorch on CPU.** PyTorch calls oneDNN, which has
  years of vectorised, cache-blocked convolution code. The C++ CPU engine exists as a
  readable reference that proves the maths matches; the GPU is where hand-written kernels
  get interesting.

## How the CUDA engine works

`cpp/src/cuda_engine.cu`, four kernels, one batch at a time:

| Stage | Kernel | Idea |
|---|---|---|
| conv1 + ReLU | `conv3x3_relu<16>` | Each block computes a 16×16 output patch of one channel. The 18×18 input tile and the channel's weights go into shared memory, so each input value is read from global memory once per block instead of nine times. ReLU is applied before the write. |
| conv2 + ReLU + pool | `conv3x3_relu_pool2<8>` | **Fused.** Each thread computes the four conv outputs under one 2×2 window and writes only their max. The 24×24×32 activation map (74 KB per image) never touches global memory. |
| fc1 + ReLU | `linear<16>` | Tiled matrix multiply: 16×16 tiles of the input and weights in shared memory, with one padding column to avoid bank conflicts. |
| fc2 | `linear<16>` | Same kernel. |

Flatten costs nothing: the pooled output is already `[C, H, W]` in memory, which is
exactly what `torch.flatten` produces.

Why fusion helps: without it, conv2 writes 74 KB per image to GPU memory and the pool
kernel reads it straight back. At batch 256 that is about 38 MB of extra memory traffic
per batch, for a layer whose arithmetic is cheap. Small CNNs are usually limited by memory
bandwidth, not compute.

Compiled with `nvcc -Xptxas -v` for sm_86: no register spills in any kernel.

## Experiments to run on your GPU

```bat
:: 1. Baseline, with a per-stage timing breakdown (CUDA events)
cpp\build\Release\mnist_infer.exe --artifacts artifacts --engine cuda --profile

:: 2. Measure the fusion: same run with conv2 and max pool as separate kernels
cpp\build\Release\mnist_infer.exe --artifacts artifacts --engine cuda --profile --no-fusion

:: 3. Batch size: latency vs throughput
for %b in (1 8 32 256 1024) do cpp\build\Release\mnist_infer.exe --artifacts artifacts --engine cuda --batch %b
```

Things to look for:

- With `--profile`, compare **"copy in"** with the kernel times. For a model this small,
  copying images over PCIe can cost more than running the network. The usual fixes are
  pinned (page-locked) host memory and overlapping copies with compute across streams.
- Compare the hand-written engine with `python benchmark.py --device cuda`. PyTorch uses
  cuDNN, and **CUDA Graphs** will show how much of PyTorch's batch-1 time is launch
  overhead, not GPU work.
- fp16 in the PyTorch benchmark gives a first look at reduced precision on the GPU.

## Debugging kernels in Visual Studio (Nsight Visual Studio Edition)

Nsight Visual Studio Edition installs with the CUDA Toolkit when Visual Studio is present.
It adds an **Extensions → Nsight** menu (just **Nsight** in some versions) and lets you put
breakpoints *inside* CUDA kernels. Generate a Visual Studio solution, in its own build
folder so it doesn't clash with CLion's:

```bat
cmake -S cpp -B cpp\build-vs -G "Visual Studio 17 2022" -A x64
start cpp\build-vs\mnist_infer.sln
```

1. Right-click **mnist_infer**, choose **Set as Startup Project**, and select the **Debug**
   configuration. Debug builds compile kernels with `-G` (device debug info); Release builds
   use `-O3 -lineinfo` for profiling.
2. In **Properties → Debugging**, set **Command Arguments** to
   `--artifacts ..\..\artifacts --engine cuda --batch 8 --repeat 1`.
   Use a small batch, because `-G` kernels are much slower.
3. Put a breakpoint inside `conv3x3_relu_pool2` in `cuda_engine.cu`, then use
   **Nsight → Start CUDA Debugging (Next-Gen)**. Plain F5 only debugs the host code.
4. When it stops, look at `threadIdx`, `blockIdx`, `acc[]` and the shared-memory `tile` in
   the Watch/Locals windows, and use **Nsight → Windows → Warp Info** to switch between
   threads. A breakpoint condition such as
   `threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.z == 0` stops on one thread only.

Check for memory errors and shared-memory races (for example a missing `__syncthreads()`)
with Compute Sanitizer, which comes with the toolkit:

```bat
compute-sanitizer --tool memcheck  cpp\build-vs\Debug\mnist_infer.exe --artifacts artifacts --engine cuda --batch 8 --repeat 1
compute-sanitizer --tool racecheck cpp\build-vs\Debug\mnist_infer.exe --artifacts artifacts --engine cuda --batch 8 --repeat 1
```

## Profiling with Nsight

Profile **Release** builds only: Debug kernels with `-G` are unoptimised, so their timings
mean nothing. The build uses `-lineinfo`, so the profilers can map time back to source lines:

```bat
nsys profile -o timeline cpp\build\Release\mnist_infer.exe --artifacts artifacts --engine cuda
ncu --set full -o kernels cpp\build\Release\mnist_infer.exe --artifacts artifacts --engine cuda --repeat 1
```

Open `timeline.nsys-rep` in Nsight Systems to see copies and kernels on a timeline, and
`kernels.ncu-rep` in Nsight Compute for each kernel's memory throughput, occupancy and
whether it is memory- or compute-bound. Both are standalone apps; you can also launch the
program from their GUIs (Nsight Compute: **Connect → Launch**, with the same arguments).

Worth looking at in Nsight Compute:

- **GPU Speed Of Light**: whether each kernel is limited by memory or by compute.
- **Memory Workload Analysis** for `conv3x3_relu_pool2` against the unfused pair
  (`--no-fusion`): how many fewer bytes the fused kernel moves to and from DRAM.
- **Shared memory bank conflicts** in `linear`: change `ws[TILE][TILE + 1]` to
  `ws[TILE][TILE]`, rebuild, and watch the conflicts appear.
- The **Source** page: which lines of each kernel take the time.

## Artifact formats

Both binary files are little-endian. `python/export.py` documents the layout, and
`cpp/src/model_io.cpp` reads it:

- `mnist_cnn.bin`: `"MNCN"`, version, tensor count, then per tensor: name, shape, float32 data.
- `mnist_test.bin`: `"MNTS"`, version, sizes, uint8 labels, float32 normalised images, and
  PyTorch's float32 logits for every image (the reference the C++ engine is checked against).

## Next steps

- Pinned host memory and two CUDA streams, so copies overlap with compute.
- CUDA Graphs in the C++ engine (`cudaStreamBeginCapture`) to remove launch overhead at batch 1.
- fp16 kernels, then Tensor Cores (`wmma`) for conv2 written as an implicit GEMM.
- Load `mnist_cnn.onnx` into TensorRT and compare it with the hand-written kernels.
