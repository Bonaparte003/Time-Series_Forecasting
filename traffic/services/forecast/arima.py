import time

import numpy as np
import pandas as pd
from pmdarima import auto_arima

from traffic.services.forecast.base import BaseForecaster, ForecastResult


class ArimaForecaster(BaseForecaster):
    name = "arima"

    def __init__(
        self,
        seq_len: int | None = None,
        *,
        seasonal_m: int = 144,
        max_p: int = 3,
        max_q: int = 3,
        max_P: int = 2,
        max_Q: int = 2,
        n_fits: int = 8,
    ):
        from django.conf import settings

        super().__init__(
            seq_len=seq_len,
            seasonal_m=seasonal_m or settings.SEASONAL_PERIOD,
            max_p=max_p,
            max_q=max_q,
            max_P=max_P,
            max_Q=max_Q,
            n_fits=n_fits,
        )
        self.seasonal_m = seasonal_m
        self.max_p = max_p
        self.max_q = max_q
        self.max_P = max_P
        self.max_Q = max_Q
        self.n_fits = n_fits

    def fit_predict(
        self,
        train: pd.Series,
        test: pd.Series,
        *,
        verbose: bool = True,
        save_training_curve: bool = True,
        curve_dir=None,
        square_id: int | None = None,
    ) -> ForecastResult:
        if verbose:
            print(
                f"  [{self.name}] auto_arima m={self.seasonal_m} "
                f"max_p={self.max_p} max_q={self.max_q} n_fits={self.n_fits}"
            )
        t0 = time.perf_counter()
        model = auto_arima(
            train.values,
            seasonal=True,
            m=self.seasonal_m,
            suppress_warnings=True,
            error_action="ignore",
            max_p=self.max_p,
            max_q=self.max_q,
            max_P=self.max_P,
            max_Q=self.max_Q,
            n_fits=self.n_fits,
            stepwise=True,
        )
        train_seconds = time.perf_counter() - t0

        preds = []
        t1 = time.perf_counter()
        for actual in test.values:
            fc = model.predict(n_periods=1)
            preds.append(float(fc[0]))
            model.update(actual)
        predict_seconds = time.perf_counter() - t1

        return ForecastResult(
            model_name=self.name,
            y_true=test.values,
            y_pred=np.array(preds),
            train_seconds=train_seconds,
            predict_seconds=predict_seconds,
            index=test.index,
            hyperparams=self.hyperparams,
        )
