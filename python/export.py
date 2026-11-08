"""
Export the trained model into the artifacts the inference engines consume:

    artifacts/mnist_cnn.bin       weights in a small binary format read by the C++/CUDA engine
    artifacts/mnist_test.bin      the 10k test images, labels and PyTorch's logits for each
                                  (so the C++ engine can check it matches PyTorch exactly)
    artifacts/mnist_cnn.onnx      ONNX graph for ONNX Runtime / TensorRT
    artifacts/mnist_cnn_ts.pt     TorchScript, loadable from C++ with libtorch

Binary layout (little-endian), weights file:
    char[4] "MNCN", u32 version=1, u32 tensor_count
    per tensor: u32 name_len, name bytes, u32 ndim, u32 dims[ndim], f32 data[prod(dims)]

Test file:
    char[4] "MNTS", u32 version=1, u32 count, u32 rows, u32 cols, u32 classes
    u8 labels[count], f32 images[count*rows*cols] (normalised), f32 logits[count*classes]
"""

import os
import struct

import numpy as np
import torch

from data import test_tensors
from model import Net, load

ROOT = os.path.join(os.path.dirname(__file__), "..")
ARTIFACTS = os.path.join(ROOT, "artifacts")


def write_weights(model: Net, path: str):
    state = model.state_dict()
    with open(path, "wb") as f:
        f.write(b"MNCN" + struct.pack("<II", 1, len(state)))
        for name, t in state.items():
            data = t.detach().cpu().contiguous().numpy().astype("<f4")
            encoded = name.encode("utf-8")
            f.write(struct.pack("<I", len(encoded)) + encoded)
            f.write(struct.pack("<I", data.ndim) + struct.pack(f"<{data.ndim}I", *data.shape))
            f.write(data.tobytes())


def write_test_set(model: Net, path: str):
    images, labels = test_tensors()
    with torch.no_grad():
        logits = torch.cat([model(images[i:i + 1000]) for i in range(0, len(images), 1000)])
    with open(path, "wb") as f:
        f.write(b"MNTS" + struct.pack("<IIIII", 1, len(images), 28, 28, logits.shape[1]))
        f.write(labels.numpy().astype(np.uint8).tobytes())
        f.write(images.numpy().astype("<f4").tobytes())
        f.write(logits.numpy().astype("<f4").tobytes())
    acc = (logits.argmax(1) == labels).float().mean().item()
    print(f"PyTorch reference accuracy on the test set: {acc:.4f}")


def main():
    model = load(os.path.join(ROOT, "models", "mnist_cnn.pt"))
    os.makedirs(ARTIFACTS, exist_ok=True)
    example = torch.randn(1, 1, 28, 28)

    write_weights(model, os.path.join(ARTIFACTS, "mnist_cnn.bin"))
    write_test_set(model, os.path.join(ARTIFACTS, "mnist_test.bin"))

    torch.onnx.export(model, (example,), os.path.join(ARTIFACTS, "mnist_cnn.onnx"),
                      input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, dynamo=False)

    traced = torch.jit.freeze(torch.jit.trace(model, example))
    traced.save(os.path.join(ARTIFACTS, "mnist_cnn_ts.pt"))

    for name in sorted(os.listdir(ARTIFACTS)):
        print(f"{name:20s} {os.path.getsize(os.path.join(ARTIFACTS, name)) / 1e6:8.2f} MB")


if __name__ == "__main__":
    main()
