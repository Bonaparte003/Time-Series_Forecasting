"""Build forecasters from model name and hyperparameter dict."""

from __future__ import annotations

from traffic.services.forecast import FORECASTERS


def default_params(model_name: str) -> dict:
    from django.conf import settings

    return dict(settings.DEFAULT_HYPERPARAMS.get(model_name, {}))


def create_forecaster(model_name: str, params: dict | None = None):
    cls = FORECASTERS[model_name]
    merged = {**default_params(model_name), **(params or {})}
    return cls(**merged)
