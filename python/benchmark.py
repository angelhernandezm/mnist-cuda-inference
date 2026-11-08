"""
Inference benchmark: the same trained model served several ways, compared on
latency, throughput and accuracy.

    python benchmark.py                 # CPU variants
    python benchmark.py --device cuda   # GPU variants (fp32, fp16, CUDA Graphs, ...)

Writes results/benchmark_<device>.md.
"""

import argparse
import os
import platform
import statistics
import time
import warnings
from typing import Callable

import torch

from data import test_tensors
from model import load

ROOT = os.path.join(os.path.dirname(__file__), "..")
ARTIFACTS = os.path.join(ROOT, "artifacts")
BATCH_SIZES = [1, 32, 256]


def timed(fn: Callable, x: torch.Tensor, sync: Callable, min_time: float = 1.0, max_iters: int = 300):
    """Per-call latencies in ms, after warm-up."""
    for _ in range(10):
        fn(x)
    sync()
    samples, start = [], time.perf_counter()
    while len(samples) < max_iters and (time.perf_counter() - start < min_time or len(samples) < 20):
        t = time.perf_counter()
        fn(x)
        sync()
        samples.append((time.perf_counter() - t) * 1e3)
    return samples


def accuracy(fn: Callable, images: torch.Tensor, labels: torch.Tensor, batch: int = 256) -> float:
    correct = 0
    for i in range(0, len(images), batch):
        correct += (fn(images[i:i + batch]).float().cpu().argmax(1) == labels[i:i + batch]).sum().item()
    return correct / len(images)


def cpu_variants(model):
    variants = {"eager fp32": model}

    ts = torch.jit.load(os.path.join(ARTIFACTS, "mnist_cnn_ts.pt"))
    variants["TorchScript (frozen)"] = ts

    try:
        # Deprecated in favour of torchao, but still the simplest int8 demo with no extra dependency
        from torch.ao.quantization import quantize_dynamic
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            variants["dynamic int8 (Linear layers)"] = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    except Exception as e:
        print(f"skipping int8: {e}")

    try:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(os.path.join(ARTIFACTS, "mnist_cnn.onnx"), opts,
                                    providers=["CPUExecutionProvider"])
        variants["ONNX Runtime"] = lambda x: torch.from_numpy(sess.run(None, {"image": x.numpy()})[0])
    except Exception as e:
        print(f"skipping ONNX Runtime: {e}")

    return variants


def cuda_variants(model):
    model = model.cuda()
    half = load(os.path.join(ROOT, "models", "mnist_cnn.pt"), "cuda").half()
    variants = {
        "CUDA eager fp32": lambda x: model(x.cuda()),
        "CUDA eager fp16": lambda x: half(x.cuda().half()),
    }

    # CUDA Graphs: record the kernel launches once per batch size, then replay them.
    # This removes Python and launch overhead, which dominates at small batch sizes.
    graphs = {}

    def graphed(x):
        n = x.shape[0]
        if n not in graphs:
            static_in = torch.zeros(n, 1, 28, 28, device="cuda")
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                for _ in range(3):
                    model(static_in)
            torch.cuda.current_stream().wait_stream(stream)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                static_out = model(static_in)
            graphs[n] = (g, static_in, static_out)
        g, static_in, static_out = graphs[n]
        static_in.copy_(x, non_blocking=True)
        g.replay()
        return static_out

    variants["CUDA Graphs fp32"] = graphed
    return variants


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")

    model = load(os.path.join(ROOT, "models", "mnist_cnn.pt"))
    images, labels = test_tensors()

    if args.device == "cuda":
        variants = cuda_variants(model)
        sync = torch.cuda.synchronize
        hw = torch.cuda.get_device_name()
        images_dev = images.cuda()
    else:
        variants = cpu_variants(model)
        sync = lambda: None
        hw = f"{platform.processor() or platform.machine()}, {torch.get_num_threads()} threads"
        images_dev = images

    rows = []
    with torch.inference_mode():
        for name, fn in variants.items():
            acc = accuracy(fn, images_dev, labels)
            cells = []
            for bs in BATCH_SIZES:
                lat = timed(fn, images_dev[:bs], sync)
                p50 = statistics.median(lat)
                p99 = sorted(lat)[int(0.99 * (len(lat) - 1))]
                cells.append(f"{p50:.3f} / {p99:.3f} / {bs / (statistics.mean(lat) / 1e3):,.0f}")
            rows.append((name, f"{acc:.4f}", *cells))
            print(f"{name:30s} acc {acc:.4f}  " + "  |  ".join(cells))

    header = ["Variant", "Accuracy"] + [f"batch {bs}: p50 ms / p99 ms / img/s" for bs in BATCH_SIZES]
    lines = [f"# Benchmark: {args.device} ({hw})", "", f"torch {torch.__version__}", "",
             "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]

    out_dir = os.path.join(ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"benchmark_{args.device}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
