"""
Django settings for time_series_forecasting project.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = "django-insecure-formative-dev-only-change-in-production"
DEBUG = True
ALLOWED_HOSTS = ["localhost", "127.0.0.1"]

INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.staticfiles",
    "traffic.apps.TrafficConfig",
]

MIDDLEWARE = []
ROOT_URLCONF = "time_series_forecasting.urls"
WSGI_APPLICATION = "time_series_forecasting.wsgi.application"
ASGI_APPLICATION = "time_series_forecasting.asgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
USE_TZ = True
TIME_ZONE = "Europe/Rome"
LANGUAGE_CODE = "en-us"

STATIC_URL = "static/"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
            ],
        },
    },
]

# --- Project-specific paths ---
RAW_DATA_GLOB = "dataverse_files*/sms-call-internet-mi-*.txt"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "outputs"
PARQUET_PATH = PROCESSED_DIR / "internet_traffic.parquet"
SERIES_DIR = PROCESSED_DIR / "series"
METADATA_PATH = PROCESSED_DIR / "metadata.json"

# TIM Milan dataset field mapping (Barlacchi et al. field-order correction)
COUNTRY_ITALY = 39
INTERNET_COL = 7
CHUNK_SIZE = 500_000

# 10-minute intervals
INTERVALS_PER_DAY = 144
SEASONAL_PERIOD = INTERVALS_PER_DAY

# Task 2 / Task 3 areas
FIXED_SQUARE_IDS = [4159, 4556]
TEST_START = "2013-12-16"
TEST_END = "2013-12-22"
VAL_START = "2013-12-09"
VAL_END = "2013-12-15"
EDA_TWO_WEEKS_END = "2013-11-14"

# Forecasting defaults
SEQUENCE_LENGTH = 144
FORECAST_MODELS = ("ets", "lstm", "tcn")

# Neural training: hold out last N intervals of train series for early stopping
NN_TRAIN_HOLDOUT_INTERVALS = 144
NN_EARLY_STOPPING_PATIENCE = 3
NN_EARLY_STOPPING_MIN_DELTA = 1e-5

EXPERIMENTS_DIR = OUTPUT_DIR / "experiments"
BEST_PARAMS_PATH = PROCESSED_DIR / "best_hyperparams.json"
FAILURE_DIR = OUTPUT_DIR / "failure_analysis"

# ETS (Holt–Winters): single fit via statsmodels; low RAM vs seasonal ARIMA search.
DEFAULT_HYPERPARAMS = {
    "ets": {
        "seasonal_periods": 144,
        "trend": None,
        "seasonal": "add",
        "damped_trend": False,
    },
    "lstm": {
        "seq_len": 144,
        "epochs": 15,
        "batch_size": 64,
        "lr": 1e-3,
        "hidden": 64,
        "layers": 2,
        "early_stopping": True,
        "patience": 3,
        "holdout_intervals": 144,
        "min_delta": 1e-5,
    },
    "tcn": {
        "seq_len": 144,
        "epochs": 15,
        "batch_size": 64,
        "lr": 1e-3,
        "channels": 32,
        "early_stopping": True,
        "patience": 3,
        "holdout_intervals": 144,
        "min_delta": 1e-5,
    },
}

# Phase 2 grid search (validation week only; test week never used here)
EXPERIMENT_GRIDS = {
    "ets": {
        "trend": [None, "add"],
        "damped_trend": [False, True],
    },
    "lstm": {
        "seq_len": [72, 144],
        "hidden": [32, 64],
        "epochs": [10, 15],
        "lr": [1e-3, 5e-4],
    },
    "tcn": {
        "seq_len": [72, 144],
        "channels": [16, 32],
        "epochs": [10, 15],
        "lr": [1e-3, 5e-4],
    },
}

# Smaller grid for smoke tests (--quick)
EXPERIMENT_GRIDS_QUICK = {
    "ets": {"trend": [None, "add"]},
    "lstm": {"seq_len": [72, 144], "epochs": [10]},
    "tcn": {"seq_len": [72, 144], "epochs": [10]},
}

EXPERIMENT_PHASES = [
    {
        "phase": 1,
        "name": "baseline",
        "use_grid": False,
        "reasoning": (
            "Establish baseline performance with literature-inspired defaults "
            "(daily seasonality period 144 for ETS/NNs, sequence length one day for NNs)."
        ),
    },
    {
        "phase": 2,
        "name": "grid_search",
        "use_grid": True,
        "reasoning": (
            "Systematic grid search on validation week (Dec 9–15) to tune "
            "capacity and learning rate without touching the held-out test week."
        ),
    },
]
