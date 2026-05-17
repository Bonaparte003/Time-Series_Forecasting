# Milan Mobile Network Traffic — Time Series Forecasting

Django project **`time_series_forecasting`** for **Formative 1** (Comparative Time Series Analysis and Forecasting of Mobile Network Traffic), using the TIM Milan telecommunications dataset.

| Name | Module |
|------|--------|
| Django project | `time_series_forecasting` |
| Django app | `traffic` |

## Requirements

- Python 3.10–3.12 (PyTorch wheels may be unavailable on 3.14+)
- ~8 GB RAM recommended for full ingest (dataset ~5 GB raw)
- macOS, Linux, or Windows
- `numpy>=1.26,<2` (required for PyTorch compatibility; pinned in `requirements.txt`)

## Setup

```bash
cd Time-Series_Forecasting
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
```

Place raw `.txt` files in the repo root under `dataverse_files/`, `dataverse_files-2/`, … `dataverse_files-7/` (already present if you cloned with data).

## Pipeline (run in order)

### 1. Task 1 — Ingest raw data

Chunked read, Italy-only (`country_code=39`), internet column, optimized dtypes, daily Parquet shards:

```bash
python manage.py ingest_raw
```

Step-by-step progress on any pipeline command:

```bash
python manage.py ingest_raw --verbose
python manage.py build_series --verbose
python manage.py run_eda --verbose
python manage.py run_pipeline --verbose
# or Django verbosity 2+ (use `-v 2`, not bare `-v`):
python manage.py build_series -v 2
```

`run_experiments` and `run_forecast` log by default; use `--quiet` to suppress.

Quick test (first 2 days only):

```bash
python manage.py ingest_raw --max-files 2
```

Memory before/after report: `data/processed/ingest_report.json`

### 2. Build per-square series

```bash
python manage.py build_series
```

Writes:

- `data/processed/series/square_<id>.parquet` for top-traffic square + squares 4159 and 4556
- `data/processed/metadata.json` (includes `top_traffic_square_id`)

### 3. Task 2 — Exploratory analysis

```bash
python manage.py run_eda
```

Figures under `data/outputs/eda/`.

### 4. Hyperparameter experiments (validation week Dec 9–15)

```bash
python manage.py run_experiments
python manage.py run_experiments --quick   # smaller grid for testing
```

ARIMA defaults use a **reduced search space** (`n_fits=3`, lower order caps) to limit RAM on laptops. Full grid: **37** experiment runs (5 ARIMA + 16 LSTM + 16 TCN on the tuning square); `--quick`: **8** runs.

Writes `data/outputs/experiments/experiments_log.csv`, `experiment_journal.md`, and `data/processed/best_hyperparams.json`.

### 5. Task 3 — Forecasting (ARIMA + LSTM + TCN)

Test week: **2013-12-16 → 2013-12-22** (held out until this step).

```bash
python manage.py run_forecast
```

Uses tuned hyperparameters from `best_hyperparams.json` (override with `--no-best-params`).

### 6. Failure analysis (worst intervals + residuals)

```bash
python manage.py run_failure_analysis
```

Outputs:

- `data/outputs/forecast/*.png` — 9 overlay plots
- `data/outputs/forecast/training/*.png` — LSTM/TCN loss vs epoch (6 curves)
- `data/outputs/forecast/timing_table.csv` — train/predict times + hardware JSON
- `data/outputs/experiments/experiments_log.csv` — hyperparameter experiment log
- `data/outputs/failure_analysis/` — residual plots + `failure_analysis.md`
- `data/outputs/forecast/metrics_square_*.csv` — MAE, MAPE, RMSE tables
- SQLite `ForecastRun` records

### View plots and training curves in the browser

```bash
python manage.py runserver
```

- http://127.0.0.1:8000/ — dashboard with all PNG galleries  
- http://127.0.0.1:8000/guide/ — full project explanation  

`run_forecast` prints **per-epoch loss** for LSTM/TCN in the terminal and saves curves under `data/outputs/forecast/training/`.

## Project layout

```
time_series_forecasting/   Django project (settings, urls, wsgi)
traffic/                   Django app
  services/       loader, ETL, EDA, forecast models
  management/commands/
data/processed/   Parquet + series (generated)
data/outputs/     Figures and metrics (generated)
```

## Models

| Model | Type | Notes |
|-------|------|--------|
| `arima` | Statistical | `pmdarima.auto_arima`, seasonal period 144 (10 min) |
| `lstm` | Neural | 2-layer LSTM, sequence length 144 |
| `tcn` | Neural | Temporal convolution network |

## Submission subset

For grading without the full 5 GB corpus, include:

- Source code (this repo)
- `data/processed/series/` for the three target squares
- `data/processed/metadata.json`
- Instructions above

Re-run `run_forecast` on the subset after `build_series`.

## References

- Barlacchi et al., *Scientific Data* 2, 150055 (2015)
- [Telecommunications dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV)
- [Grid dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/QJWLFU)

## AI use

Document any AI assistance in your PDF report per course integrity guidelines.
