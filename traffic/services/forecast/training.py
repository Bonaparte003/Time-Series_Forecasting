"""Shared PyTorch training loop with early stopping and per-epoch metrics."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def get_torch_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainingHistory:
    """Per-epoch training diagnostics (one list entry per completed epoch)."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_mae: list[float] = field(default_factory=list)
    val_rmse: list[float] = field(default_factory=list)
    best_epoch: int = 0
    stopped_early: bool = False
    max_epochs: int = 0

    @property
    def epochs_run(self) -> int:
        return len(self.train_loss)

    def to_dict(self) -> dict:
        return asdict(self)


def split_sequences_train_val(
    X: np.ndarray,
    y: np.ndarray,
    holdout_intervals: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Hold out the last `holdout_intervals` sequences for validation (time-ordered)."""
    n = len(X)
    if holdout_intervals <= 0 or holdout_intervals >= n:
        raise ValueError(
            f"holdout_intervals must be in 1..{n - 1}, got {holdout_intervals} (n={n})"
        )
    split = n - holdout_intervals
    return X[:split], y[:split], X[split:], y[split:]


def _align_target_shape(pred: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
    """LSTM returns (batch, 1); TCN returns (batch,)."""
    if pred.shape != yb.shape:
        return pred.squeeze(-1)
    return pred


def _prediction_scalar(pred: torch.Tensor) -> float:
    """Single-step model output → float (handles LSTM (1,1) and TCN (1,))."""
    return float(pred.reshape(-1)[0].item())


@torch.no_grad()
def one_step_ahead_predict(
    model: nn.Module,
    scaler,
    scaled_train: np.ndarray,
    test: pd.Series,
    seq_len: int,
    device: torch.device,
) -> np.ndarray:
    """
  One-step-ahead on the test week (assignment Task 3).

  At each test time t, the input window uses true scaled traffic up to and
  including t; the model predicts the value at t+1 (stored as test[t]).
  """
    model.eval()
    test_scaled = scaler.transform(test.values.reshape(-1, 1)).flatten()
    history = list(scaled_train)
    preds_scaled: list[float] = []

    for i in range(len(test)):
        window = torch.tensor(
            history[-seq_len:], dtype=torch.float32, device=device
        ).view(1, seq_len, 1)
        preds_scaled.append(_prediction_scalar(model(window)))
        history.append(test_scaled[i])

    return scaler.inverse_transform(
        np.array(preds_scaled, dtype=float).reshape(-1, 1)
    ).flatten()


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
) -> tuple[float, float, float]:
    """Return average MSE loss, MAE, and RMSE on scaled targets."""
    model.eval()
    losses: list[float] = []
    abs_errors: list[float] = []
    sq_errors: list[float] = []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = _align_target_shape(model(xb), yb)
        losses.append(float(loss_fn(pred, yb).item()))
        diff = pred - yb
        abs_errors.extend(torch.abs(diff).detach().cpu().reshape(-1).tolist())
        sq_errors.extend((diff ** 2).detach().cpu().reshape(-1).tolist())

    if not losses:
        return float("nan"), float("nan"), float("nan")

    mse = sum(losses) / len(losses)
    mae = sum(abs_errors) / len(abs_errors)
    rmse = float(np.sqrt(sum(sq_errors) / len(sq_errors)))
    return mse, mae, rmse


def train_epochs(
    model: nn.Module,
    train_loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    device: torch.device,
    val_loader: DataLoader | None = None,
    patience: int = 3,
    min_delta: float = 1e-5,
    verbose: bool = True,
) -> TrainingHistory:
    """
    Train up to `epochs` with optional early stopping on validation MSE.

    When `val_loader` is provided, restores weights from the best validation epoch.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history = TrainingHistory(max_epochs=epochs)

    best_val = float("inf")
    best_epoch = 0
    best_state: dict | None = None
    wait = 0
    use_early_stop = val_loader is not None and patience > 0

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses: list[float] = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = _align_target_shape(model(xb), yb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        train_loss = sum(batch_losses) / len(batch_losses)
        history.train_loss.append(train_loss)

        if val_loader is not None:
            val_loss, val_mae, val_rmse = evaluate_loader(
                model, val_loader, device, loss_fn
            )
            history.val_loss.append(val_loss)
            history.val_mae.append(val_mae)
            history.val_rmse.append(val_rmse)

            improved = val_loss < best_val - min_delta
            if improved:
                best_val = val_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            elif use_early_stop:
                wait += 1

            if verbose:
                best_mark = " *best*" if improved else ""
                print(
                    f"    Epoch {epoch:>2}/{epochs}  "
                    f"train_loss={train_loss:.6f}  "
                    f"val_loss={val_loss:.6f}  "
                    f"val_mae={val_mae:.6f}  "
                    f"val_rmse={val_rmse:.6f}{best_mark}"
                )

            if use_early_stop and wait >= patience:
                history.stopped_early = True
                if verbose:
                    print(
                        f"    Early stopping at epoch {epoch} "
                        f"(best epoch {best_epoch}, patience={patience})"
                    )
                break
        elif verbose:
            print(f"    Epoch {epoch:>2}/{epochs}  train_loss={train_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        history.best_epoch = best_epoch
    else:
        history.best_epoch = len(history.train_loss)
        if val_loader is not None and verbose:
            print("    Warning: no validation improvement; using final epoch weights.")

    return history


def build_sequence_loaders(
    X: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    holdout_intervals: int,
    lstm_targets: bool,
) -> tuple[DataLoader, DataLoader | None, int, int]:
    """
    Split sequences, build train/val loaders.
    Returns (train_loader, val_loader, n_train_seq, n_val_seq).
    """
    min_train = max(batch_size, 32)
    holdout = min(holdout_intervals, max(1, len(X) // 5))
    if len(X) - holdout < min_train:
        holdout = max(1, len(X) - min_train)

    X_tr, y_tr, X_val, y_val = split_sequences_train_val(X, y, holdout)

    if lstm_targets:
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(-1)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1)
    else:
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32).unsqueeze(-1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)

    train_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader, len(X_tr), len(X_val)


def save_training_curve(
    history: TrainingHistory | list[float],
    *,
    model_name: str,
    square_id: int,
    out_dir: Path,
    suffix: str = "",
) -> Path:
    """Save loss/metric-vs-epoch PNG and JSON history."""
    if isinstance(history, list):
        history = TrainingHistory(train_loss=history, max_epochs=len(history))

    out_dir.mkdir(parents=True, exist_ok=True)
    extra = f"_{suffix}" if suffix else ""
    stem = f"training_{model_name}_square_{square_id}{extra}"
    epochs = list(range(1, history.epochs_run + 1))

    payload = {
        "model": model_name,
        "square_id": square_id,
        "epochs": epochs,
        "train_loss": history.train_loss,
        "val_loss": history.val_loss,
        "val_mae": history.val_mae,
        "val_rmse": history.val_rmse,
        "best_epoch": history.best_epoch,
        "stopped_early": history.stopped_early,
        "max_epochs": history.max_epochs,
    }

    json_path = out_dir / f"{stem}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history.train_loss, marker="o", label="train MSE", linewidth=1.5)
    if history.val_loss:
        axes[0].plot(
            epochs[: len(history.val_loss)],
            history.val_loss,
            marker="s",
            label="val MSE",
            linewidth=1.5,
        )
    if history.best_epoch:
        axes[0].axvline(history.best_epoch, color="green", ls="--", alpha=0.6, label="best epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE (scaled)")
    axes[0].set_title(f"{model_name.upper()} loss — square {square_id}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if history.val_mae:
        axes[1].plot(
            epochs[: len(history.val_mae)],
            history.val_mae,
            marker="o",
            color="C2",
            label="val MAE",
            linewidth=1.5,
        )
        axes[1].plot(
            epochs[: len(history.val_rmse)],
            history.val_rmse,
            marker="s",
            color="C3",
            label="val RMSE",
            linewidth=1.5,
        )
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Error (scaled)")
        axes[1].set_title("Validation MAE / RMSE")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].set_visible(False)

    fig.suptitle(
        f"{'Early stop' if history.stopped_early else 'Full run'} "
        f"— {history.epochs_run}/{history.max_epochs} epochs",
        fontsize=10,
    )
    fig.tight_layout()
    png_path = out_dir / f"{stem}.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return png_path
