# Directory and file reference

Where everything lives, what generates it, and what to commit for grading. Paths are relative to the repository root unless noted.

**See also:** [Methods reference](METHODS.md) · [Project guide](../PROJECT_GUIDE.md) · [README](../README.md)

---

## Top-level layout

```
Time-Series_Forecasting/
├── README.md                      # Quick start, gallery, metrics summary
├── PROJECT_GUIDE.md               # Pipeline, commands, assignment mapping
├── requirements.txt               # Python dependencies
├── manage.py                      # Django CLI entry point
├── db.sqlite3                     # SQLite (experiment/forecast run logs) — local only
│
├── time_series_forecasting/       # Django project package
│   ├── settings.py                # Paths, date splits, hyperparameter grids
│   ├── urls.py                    # Web routes (presentation UI, outputs, docs/images)
│   ├── wsgi.py / asgi.py
│
├── traffic/                       # Main application
│   ├── models.py                  # ExperimentRun, ForecastRun, IngestionRun
│   ├── views.py                   # Presentation page + image serving
│   ├── migrations/                # Database schema
│   ├── management/commands/       # `python manage.py <command>`
│   └── services/                  # All pipeline logic (no Django views here)
│
├── docs/
│   ├── METHODS.md                 # Model & evaluation methodology (this doc’s sibling)
│   ├── DIRECTORY_REFERENCE.md     # You are here
│   └── images/                    # Committed PNGs for README / GitHub rendering
│       ├── eda/
│       ├── forecast/
│       ├── training/
│       └── failure/
│
├── data/
│   ├── processed/                 # Intermediate Parquet & metadata (mostly gitignored)
│   └── outputs/                   # Plots, metrics, logs (gitignored)
│
└── dataverse_files*/              # Raw TIM downloads (gitignored) — place files here
```

---

## Django project — `time_series_forecasting/`

| File | Purpose |
|------|---------|
| `settings.py` | `RAW_DATA_GLOB`, `PROCESSED_DIR`, `OUTPUT_DIR`, train/val/test dates, `DEFAULT_HYPERPARAMS`, `EXPERIMENT_GRIDS`, `FORECAST_MODELS` |
| `urls.py` | Routes: `/` (presentation), `/outputs/<path>`, `/docs-images/<path>` |

---

## Application — `traffic/`

### `management/commands/`

| Command | Service module | Task |
|---------|----------------|------|
| `ingest_raw` | `loader.py` | Task 1 — chunked ingest → daily Parquet |
| `build_series` | `etl.py` | Resample to 10-min series per square |
| `run_eda` | `eda.py` | Task 2 — exploratory figures |
| `run_experiments` | `experiments.py` | Hyperparameter grid on validation week |
| `run_forecast` | `forecast_runner.py` | Task 3 — final test-week evaluation |
| `run_failure_analysis` | `failure_analysis.py` | Worst windows + residual plots |
| `run_pipeline` | (chains all above) | Full end-to-end run |

Common flags: `--verbose`, `--quiet`; ingest supports `--max-files`; experiments supports `--quick`.

### `services/` — core logic

| Module | Role |
|--------|------|
| `paths.py` | Discover raw files; `ensure_dirs()` for output trees |
| `loader.py` | Chunked CSV read, dtype optimization, daily Parquet write, `ingest_report.json` |
| `etl.py` | Aggregate to 10-min UTC series; `load_square_series()`, `metadata.json` |
| `eda.py` | PDF, stationarity, decomposition, ACF/PACF, heatmap, outliers |
| `experiments.py` | Phase 1/2 runs, SQLite `ExperimentRun`, `best_hyperparams.json` |
| `forecast_runner.py` | Loop squares × models; metrics CSV; timing; forecast PNGs |
| `failure_analysis.py` | Sliding-window worst periods; `failure_analysis.md` |
| `hardware.py` | CPU/RAM/Python env JSON for timing report |
| `verbose.py` | Shared `--verbose` / `--quiet` logging helpers |

### `services/forecast/`

| Module | Role |
|--------|------|
| `base.py` | `BaseForecaster`, train/test splits, `make_sequences()` |
| `ets.py` | Holt–Winters forecaster |
| `lstm.py` | LSTM forecaster |
| `tcn.py` | TCN forecaster |
| `training.py` | PyTorch training, early stopping, `one_step_ahead_predict()` |
| `metrics.py` | MAE, MAPE, RMSE |
| `factory.py` | `create_forecaster(model_name, params)` |

### `models.py` (database)

| Model | Stores |
|-------|--------|
| `IngestionRun` | Ingest timing / memory notes |
| `ExperimentRun` | Each tuning run (params, val metrics, reasoning) |
| `ForecastRun` | Latest test-week metrics per square × model |

---

## Raw data — `dataverse_files*/`

- Pattern: `dataverse_files*/sms-call-internet-mi-YYYY-MM-DD.txt`
- Tab-separated; fields used: square id, timestamp ms, country code, internet traffic (column index 7 per Barlacchi correction).
- **Gitignored** (~5 GB total). Not required in submission if you ship processed series instead.

---

## Processed data — `data/processed/`

| Path | Created by | Contents |
|------|------------|----------|
| `daily/*.parquet` | `ingest_raw` | One day per file, optimized dtypes |
| `ingest_report.json` | `ingest_raw` | Memory before/after, row counts (Task 1 evidence) |
| `series/square_<id>.parquet` | `build_series` | Single column `traffic`, 10-min index |
| `metadata.json` | `build_series` | Top-traffic square id, target square list |
| `best_hyperparams.json` | `run_experiments` | Winning params per model |

**Gitignore note:** `daily/` and `series/*.parquet` are ignored by default. For submission, **commit** `series/` for squares **5161, 4159, 4556** plus `metadata.json` and `best_hyperparams.json` so graders can skip ingest.

---

## Generated outputs — `data/outputs/`

Regenerated by pipeline commands. Listed in `.gitignore` except when you copy figures to `docs/images/`.

### `data/outputs/eda/` — Task 2

| File | Assignment item |
|------|-----------------|
| `01_traffic_pdf.png` | PDF over 10k areas |
| `02_three_areas_two_weeks.png` | Three target areas, first two weeks |
| `03_stationarity_square_<id>.png` | Rolling mean/std |
| `03_adf_square_<id>.txt` | ADF test output |
| `04_decomposition_square_5161.png` | Trend / seasonal / residual |
| `05_acf_pacf_square_5161.png` | ACF & PACF |
| `06_spatial_heatmap.png` | Grid heatmap |
| `07_outlier_sample.csv` | Outlier candidates |

### `data/outputs/experiments/` — Tuning

| File | Purpose |
|------|---------|
| `experiments_log.csv` | All 39 runs: params, val MAE/MAPE/RMSE, reasoning |
| `experiment_journal.md` | Human-readable summary for the report |

### `data/outputs/forecast/` — Task 3

| Path | Purpose |
|------|---------|
| `{ets,lstm,tcn}_square_<id>.png` | **9** overlay plots (actual vs predicted) |
| `metrics_square_<id>.csv` | **3** tables (one per square) |
| `metrics_summary.json` | All metrics aggregated |
| `timing_table.csv` | Train/predict seconds per run |
| `timing_report.json` | Timing + environment note |
| `hardware_environment.json` | CPU, RAM, library versions |
| `predictions/<model>_square_<id>.csv` | Per-timestamp actual, predicted, residual |
| `training/training_<model>_square_<id>_*.png` | Neural net loss curves |
| `training/training_<model>_square_<id>_*.json` | Per-epoch train/val losses |

### `data/outputs/failure_analysis/` — Task 3 §VIII

| File | Purpose |
|------|---------|
| `failure_analysis.md` | Draft failure-section text |
| `failure_report.json` | Worst windows metadata |
| `residuals_<model>_square_<id>.png` | Residuals over test week |
| `worst_window_<model>_square_<id>.png` | Zoom on worst 6-hour window |

---

## Committed figures — `docs/images/`

Mirrors selected outputs for README and offline viewing. **Not** auto-synced — copy or script after a pipeline run if you update results.

```
docs/images/
├── eda/           # Task 2 gallery
├── forecast/      # 9 test-week overlays
├── training/      # Sample NN training curves
└── failure/       # Sample failure-analysis plots
```

---

## Web UI

```bash
python manage.py runserver
```

| URL | Content |
|-----|---------|
| http://127.0.0.1:8000/ | Presentation — tasks, figures, metrics, code |
| http://127.0.0.1:8000/outputs/… | PNG/JSON from `data/outputs/` |
| http://127.0.0.1:8000/docs-images/… | Fallback PNGs from `docs/images/` |

---

## Configuration quick reference

From `time_series_forecasting/settings.py`:

| Setting | Value |
|---------|-------|
| `SEASONAL_PERIOD` / `SEQUENCE_LENGTH` | 144 |
| `VAL_START` / `VAL_END` | 2013-12-09 / 2013-12-15 |
| `TEST_START` / `TEST_END` | 2013-12-16 / 2013-12-22 |
| `FIXED_SQUARE_IDS` | 4159, 4556 (+ top traffic from metadata) |
| `FORECAST_MODELS` | `ets`, `lstm`, `tcn` |
| `NN_TRAIN_HOLDOUT_INTERVALS` | 144 |

---

## Suggested reading order

1. [README.md](../README.md) — install and run pipeline  
2. [PROJECT_GUIDE.md](../PROJECT_GUIDE.md) — commands and assignment checklist  
3. [METHODS.md](METHODS.md) — models and evaluation protocol  
4. [DIRECTORY_REFERENCE.md](DIRECTORY_REFERENCE.md) — this file, when locating artifacts  
