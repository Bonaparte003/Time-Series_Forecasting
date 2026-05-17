from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
from django.conf import settings
from sklearn.preprocessing import MinMaxScaler


@dataclass
class ForecastResult:
    model_name: str
    y_true: np.ndarray
    y_pred: np.ndarray
    train_seconds: float
    predict_seconds: float
    index: pd.DatetimeIndex
    epoch_losses: list[float] | None = None
    training_history: dict | None = None
    hyperparams: dict | None = None


@dataclass
class MetricBundle:
    mae: float
    mape: float
    rmse: float


class BaseForecaster(ABC):
    name: str = "base"

    def __init__(self, seq_len: int | None = None, **kwargs):
        self.seq_len = seq_len or settings.SEQUENCE_LENGTH
        self.scaler = MinMaxScaler()
        self.hyperparams = {"seq_len": self.seq_len, **kwargs}

    def split_train_test(self, series: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Final evaluation: train before test week, hold out Dec 16–22."""
        test_start = pd.Timestamp(settings.TEST_START, tz="UTC")
        test_end = pd.Timestamp(settings.TEST_END, tz="UTC") + pd.Timedelta(days=1)
        train = series[series.index < test_start]
        test = series[(series.index >= test_start) & (series.index < test_end)]
        return train, test

    def split_for_tuning(self, series: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Hyperparameter tuning: train before validation week, validate Dec 9–15."""
        val_start = pd.Timestamp(settings.VAL_START, tz="UTC")
        test_start = pd.Timestamp(settings.TEST_START, tz="UTC")
        train = series[series.index < val_start]
        val = series[(series.index >= val_start) & (series.index < test_start)]
        return train, val

    @abstractmethod
    def fit_predict(
        self,
        train: pd.Series,
        test: pd.Series,
        *,
        verbose: bool = True,
        save_curve: bool = True,
        curve_dir=None,
        square_id: int | None = None,
    ) -> ForecastResult:
        pass

    def make_sequences(self, values: np.ndarray):
        X, y = [], []
        for i in range(self.seq_len, len(values)):
            X.append(values[i - self.seq_len : i])
            y.append(values[i])
        return np.array(X), np.array(y)
