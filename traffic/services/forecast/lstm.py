import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from traffic.services.forecast.base import BaseForecaster, ForecastResult
from traffic.services.forecast.training import (
    build_sequence_loaders,
    get_torch_device,
    save_training_curve,
    train_epochs,
)


class LSTMNet(nn.Module):
    def __init__(self, input_size=1, hidden=64, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class LstmForecaster(BaseForecaster):
    name = "lstm"

    def __init__(
        self,
        seq_len: int | None = None,
        *,
        epochs: int = 15,
        batch_size: int = 64,
        lr: float = 1e-3,
        hidden: int = 64,
        layers: int = 2,
        holdout_intervals: int | None = None,
        patience: int = 3,
        min_delta: float = 1e-5,
        early_stopping: bool = True,
    ):
        from django.conf import settings

        super().__init__(
            seq_len=seq_len,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            hidden=hidden,
            layers=layers,
            holdout_intervals=holdout_intervals
            or settings.NN_TRAIN_HOLDOUT_INTERVALS,
            patience=patience,
            min_delta=min_delta,
            early_stopping=early_stopping,
        )
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.hidden = hidden
        self.layers = layers
        self.holdout_intervals = (
            holdout_intervals or settings.NN_TRAIN_HOLDOUT_INTERVALS
        )
        self.patience = patience
        self.min_delta = min_delta
        self.early_stopping = early_stopping

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

        train_loader, val_loader, n_train, n_val = build_sequence_loaders(
            X,
            y,
            batch_size=self.batch_size,
            holdout_intervals=self.holdout_intervals,
            lstm_targets=True,
        )

        device = get_torch_device()
        model = LSTMNet(hidden=self.hidden, layers=self.layers).to(device)

        if verbose:
            print(
                f"  [{self.name}] device={device} seq_len={self.seq_len} "
                f"hidden={self.hidden} max_epochs={self.epochs} lr={self.lr} "
                f"— train_seq={n_train:,} val_seq={n_val:,} "
                f"early_stop={self.early_stopping} patience={self.patience}"
            )

        t0 = time.perf_counter()
        history = train_epochs(
            model,
            train_loader,
            epochs=self.epochs,
            lr=self.lr,
            device=device,
            val_loader=val_loader if self.early_stopping else None,
            patience=self.patience if self.early_stopping else 0,
            min_delta=self.min_delta,
            verbose=verbose,
        )
        train_seconds = time.perf_counter() - t0

        t1 = time.perf_counter()
        preds = self._walk_forward_predict(model, scaled, test, device)
        predict_seconds = time.perf_counter() - t1

        if save_training_curve and curve_dir is not None and square_id is not None:
            save_training_curve(
                history,
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
            epoch_losses=history.train_loss,
            training_history=history.to_dict(),
            hyperparams=self.hyperparams,
        )

    def _param_suffix(self) -> str:
        return f"h{self.hidden}_e{self.epochs}_s{self.seq_len}"

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
