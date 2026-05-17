"""Holt–Winters exponential smoothing (statsmodels) — lightweight statistical baseline."""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from traffic.services.forecast.base import BaseForecaster, ForecastResult


class EtsForecaster(BaseForecaster):
    name = "ets"

    def __init__(
        self,
        seq_len: int | None = None,
        *,
        seasonal_periods: int = 144,
        trend: str | None = None,
        seasonal: str | None = "add",
        damped_trend: bool = False,
        progress_every: int | None = None,
    ):
        from django.conf import settings

        super().__init__(
            seq_len=seq_len,
            seasonal_periods=seasonal_periods or settings.SEASONAL_PERIOD,
            trend=trend,
            seasonal=seasonal,
            damped_trend=damped_trend,
            progress_every=progress_every,
        )
        self.seasonal_periods = seasonal_periods or settings.SEASONAL_PERIOD
        self.trend = trend
        self.seasonal = seasonal
        self.damped_trend = damped_trend if trend is not None else False
        self.progress_every = progress_every or self.seasonal_periods

    def _log(self, verbose: bool, msg: str) -> None:
        if verbose:
            print(msg)

    def _build_model(self, endog: np.ndarray) -> ExponentialSmoothing:
        return ExponentialSmoothing(
            endog,
            trend=self.trend,
            damped_trend=self.damped_trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
            initialization_method="estimated",
        )

    def _fit(self, train: pd.Series, *, verbose: bool):
        endog = train.values.astype(float)
        self._log(
            verbose,
            f"  [{self.name}] Holt–Winters | train={len(endog):,} points | "
            f"seasonal_periods={self.seasonal_periods} trend={self.trend!r} "
            f"seasonal={self.seasonal!r} damped={self.damped_trend}",
        )
        try:
            self._log(verbose, f"  [{self.name}] optimizing parameters…")
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Optimization failed to converge",
                    category=Warning,
                )
                fitted = self._build_model(endog).fit(optimized=True)
        except Exception as exc:
            self._log(
                verbose,
                f"  [{self.name}] primary spec failed ({exc!s}); "
                "retrying with trend=None, seasonal='add'",
            )
            self.trend = None
            self.damped_trend = False
            self.hyperparams["trend"] = None
            self.hyperparams["damped_trend"] = False
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Optimization failed to converge",
                    category=Warning,
                )
                fitted = self._build_model(endog).fit(optimized=True)

        mle = getattr(fitted, "mle_retvals", None) or {}
        if verbose and mle.get("converged") is False:
            self._log(
                verbose,
                f"  [{self.name}] note: optimizer reported non-convergence "
                f"({mle.get('warnflag', '?')})",
            )

        aic = getattr(fitted, "aic", None)
        if aic is not None:
            self._log(verbose, f"  [{self.name}] fit complete | AIC={aic:.2f}")
        else:
            self._log(verbose, f"  [{self.name}] fit complete")
        return fitted

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
        t0 = time.perf_counter()
        fitted = self._fit(train, verbose=verbose)
        train_seconds = time.perf_counter() - t0

        n_test = len(test)
        y_true = test.values
        self._log(
            verbose,
            f"  [{self.name}] multi-step forecast: {n_test:,} steps "
            f"(diagnostics every {self.progress_every} intervals)",
        )

        t1 = time.perf_counter()
        fc = fitted.forecast(n_test)
        preds = np.asarray(fc, dtype=float).ravel()
        if len(preds) != n_test:
            raise ValueError(
                f"Expected {n_test} forecasts, got {len(preds)} from Holt–Winters."
            )

        if verbose:
            for step in range(1, n_test + 1):
                if step == 1 or step == n_test or step % self.progress_every == 0:
                    err = np.abs(preds[:step] - y_true[:step])
                    self._log(
                        verbose,
                        f"  [{self.name}] horizon {step:,}/{n_test:,} | "
                        f"running MAE={err.mean():.4f}",
                    )

        predict_seconds = time.perf_counter() - t1
        self._log(
            verbose,
            f"  [{self.name}] done | train={train_seconds:.1f}s "
            f"predict={predict_seconds:.1f}s",
        )

        return ForecastResult(
            model_name=self.name,
            y_true=y_true,
            y_pred=np.array(preds),
            train_seconds=train_seconds,
            predict_seconds=predict_seconds,
            index=test.index,
            hyperparams=self.hyperparams,
        )
