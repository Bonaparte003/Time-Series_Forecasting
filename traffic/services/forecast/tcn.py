import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from traffic.services.forecast.base import BaseForecaster, ForecastResult
from traffic.services.forecast.training import save_training_curve, train_epochs


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, dilation=1):
        super().__init__()
        padding = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        out = self.relu(self.conv(x))[..., : x.size(-1)]
        res = x if self.down is None else self.down(x)
        return out + res


class TCNNet(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.net = nn.Sequential(
            TCNBlock(1, channels, dilation=1),
            TCNBlock(channels, channels, dilation=2),
            TCNBlock(channels, channels, dilation=4),
        )
        self.head = nn.Linear(channels, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        h = self.net(x)
        return self.head(h[:, :, -1]).squeeze(-1)


class TcnForecaster(BaseForecaster):
    name = "tcn"

    def __init__(
        self,
        seq_len: int | None = None,
        *,
        epochs: int = 15,
        batch_size: int = 64,
        lr: float = 1e-3,
        channels: int = 32,
    ):
        super().__init__(
            seq_len=seq_len,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            channels=channels,
        )
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.channels = channels

    def fit_predict(
        self,
        train: pd.Series,
        test: pd.Series,
        *,
        verbose: bool = True,
        save_training_curve: bool = True,
        curve_dir: Path | None = None,
        square_id: int | None = None,
    ) -> ForecastResult:
        scaled = self.scaler.fit_transform(train.values.reshape(-1, 1)).flatten()
        X, y = self.make_sequences(scaled)
        if len(X) == 0:
            raise ValueError(
                f"Not enough training points for seq_len={self.seq_len} "
                f"(have {len(scaled)})."
            )
        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        y_t = torch.tensor(y, dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True
        )

        device = torch.device("cpu")
        model = TCNNet(channels=self.channels).to(device)

        if verbose:
            print(
                f"  [{self.name}] seq_len={self.seq_len} channels={self.channels} "
                f"epochs={self.epochs} lr={self.lr} — {len(X):,} sequences"
            )

        t0 = time.perf_counter()
        epoch_losses = train_epochs(
            model, loader, epochs=self.epochs, lr=self.lr, device=device, verbose=verbose
        )
        train_seconds = time.perf_counter() - t0

        t1 = time.perf_counter()
        preds = self._walk_forward_predict(model, scaled, test, device)
        predict_seconds = time.perf_counter() - t1

        if save_training_curve and curve_dir is not None and square_id is not None:
            save_training_curve(
                epoch_losses,
                model_name=self.name,
                square_id=square_id,
                out_dir=curve_dir,
                suffix=self._param_suffix(),
            )

        return ForecastResult(
            model_name=self.name,
            y_true=test.values,
            y_pred=preds,
            train_seconds=train_seconds,
            predict_seconds=predict_seconds,
            index=test.index,
            epoch_losses=epoch_losses,
            hyperparams=self.hyperparams,
        )

    def _param_suffix(self) -> str:
        return f"c{self.channels}_e{self.epochs}_s{self.seq_len}"

    def _walk_forward_predict(
        self, model: nn.Module, scaled_train: np.ndarray, test: pd.Series, device
    ) -> np.ndarray:
        model.eval()
        history = list(scaled_train)
        preds_scaled = []
        with torch.no_grad():
            for _ in range(len(test)):
                window = torch.tensor(
                    history[-self.seq_len :], dtype=torch.float32
                ).view(1, self.seq_len, 1)
                pred = model(window.to(device)).item()
                preds_scaled.append(pred)
                history.append(pred)
        return self.scaler.inverse_transform(
            np.array(preds_scaled).reshape(-1, 1)
        ).flatten()
