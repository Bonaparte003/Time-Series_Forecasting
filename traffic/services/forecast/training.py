"""Shared PyTorch training loop with per-epoch loss tracking."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def get_torch_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epochs(
    model: nn.Module,
    loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    device: torch.device,
    verbose: bool = True,
) -> list[float]:
    """
    Train for `epochs` passes over the loader.
    Returns average MSE loss per epoch (one value per epoch).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    epoch_losses: list[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses: list[float] = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        avg_loss = sum(batch_losses) / len(batch_losses)
        epoch_losses.append(avg_loss)
        if verbose:
            print(f"    Epoch {epoch:>2}/{epochs}  train_loss={avg_loss:.6f}")

    return epoch_losses


def save_training_curve(
    epoch_losses: list[float],
    *,
    model_name: str,
    square_id: int,
    out_dir: Path,
    suffix: str = "",
) -> Path:
    """Save loss-vs-epoch PNG and JSON history."""
    out_dir.mkdir(parents=True, exist_ok=True)
    extra = f"_{suffix}" if suffix else ""
    stem = f"training_{model_name}_square_{square_id}{extra}"

    json_path = out_dir / f"{stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": model_name,
                "square_id": square_id,
                "epochs": list(range(1, len(epoch_losses) + 1)),
                "train_loss": epoch_losses,
            },
            f,
            indent=2,
        )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker="o", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training MSE loss")
    ax.set_title(f"{model_name.upper()} training — square {square_id}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png_path = out_dir / f"{stem}.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return png_path
