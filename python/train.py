"""
Train the CNN on MNIST.

    python train.py --epochs 3
"""

import argparse
import os
import time

import torch
import torch.nn as nn

from data import train_val_loaders, test_tensors
from model import Net

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss += loss_fn(logits, yb).item()
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)
    return loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader = train_val_loaders(args.batch_size, args.seed)

    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        start, train_loss, seen = time.perf_counter(), 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * yb.size(0)
            seen += yb.size(0)

        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"epoch {epoch}: train_loss {train_loss / seen:.4f} | val_loss {val_loss:.4f} | "
              f"val_acc {val_acc:.4f} | {time.perf_counter() - start:.0f}s on {device}")

    images, labels = test_tensors()
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(images, labels), batch_size=512)
    _, test_acc = evaluate(model, test_loader, device)
    print(f"test accuracy (10,000 unseen images): {test_acc:.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, "mnist_cnn.pt")
    torch.save(model.cpu().state_dict(), path)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
