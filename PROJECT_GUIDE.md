# Project Guide — What This Codebase Does

## Big picture

You are analyzing **mobile internet traffic in Milan** (TIM dataset, ~10,000 grid squares, 10-minute intervals, ~2 months). The work is split into three assignment tasks, implemented as a **Django pipeline** (not a traditional CRUD web app). Django gives you:

- A standard project layout (`time_series_forecasting` + app `traffic`)
- **Management commands** to run each stage from the terminal
- A small **dashboard** (`runserver`) to browse generated plots

Heavy work is done by **pandas**, **statsmodels**, **PyTorch**, and **matplotlib** inside `traffic/services/`.

---

## Repository layout

```
time_series_forecasting/     ← Django project (settings, URLs)
traffic/                     ← Django app
  models.py                  ← SQLite records (ingest + forecast metrics)
  views.py                   ← Dashboard + image gallery
  services/
    loader.py                ← Task 1: read raw .txt in chunks
    etl.py                   ← Build 10-min time series per square
    eda.py                   ← Task 2: plots
    forecast/                ← Task 3: ARIMA, LSTM, TCN
    forecast_runner.py       ← Runs all models, saves plots & metrics
  management/commands/       ← CLI entry points
data/
  processed/                 ← Parquet + series (after ingest)
  outputs/                   ← PNG figures + CSV metrics
dataverse_files*/            ← Your raw TIM text files (7 folders)
```

---

## Data flow (run commands in this order)

### 1. `ingest_raw` — Task 1 (memory-efficient load)

**Input:** Tab-separated daily files  
`sms-call-internet-mi-YYYY-MM-DD.txt` in `dataverse_files*`

**What happens:**
- Reads files in **chunks** (500k rows) so RAM stays bounded
- Keeps only **Italy** (`country_code == 39`) and column **7** (internet traffic)
- Uses compact dtypes (`float32`, `uint16`, …)
- Writes one **Parquet file per day** under `data/processed/daily/`
- Logs **memory before/after** on a sample → `data/processed/ingest_report.json`

### 2. `build_series` — prepare time series

**What happens:**
- Loads all daily Parquet shards
- Finds the square with **highest total traffic** over the period
- Builds regular **10-minute** series for:
  - that top square
  - fixed squares **4159** and **4556**
- Saves `data/processed/series/square_<id>.parquet`
- Writes `data/processed/metadata.json` with IDs to use later

### 3. `run_eda` — Task 2 (exploratory plots)

**Outputs in `data/outputs/eda/`:**
- Traffic distribution across squares (PDF/histogram)
- First two weeks for the three target areas
- Stationarity (rolling stats + ADF text file)
- Seasonal decomposition, ACF/PACF (reference square)
- Spatial heatmap (100×100 grid)
- Sample outlier CSV

### 4. `run_experiments` — hyperparameter tuning (rubric: experimentation)

**Splits:**
- **Train:** before 2013-12-09
- **Validation:** 2013-12-09 → 2013-12-15 (tuning only)
- **Test:** 2013-12-16 → 2013-12-22 (never used until `run_forecast`)

**Phases:**
1. **Phase 1 — baseline:** default parameters, record validation MAE.
2. **Phase 2 — grid search:** combinations from `EXPERIMENT_GRIDS` in settings.

**Outputs:**
- `data/outputs/experiments/experiments_log.csv` — every run with params + metrics + reasoning
- `data/outputs/experiments/experiment_journal.md` — summary for your report
- `data/processed/best_hyperparams.json` — best validation MAE per model
- SQLite `ExperimentRun` rows

```bash
python manage.py run_experiments          # full grid (slow)
python manage.py run_experiments --quick  # smaller grid for testing
```

### 5. `run_forecast` — Task 3 (final evaluation)

**Train/test split:**
- **Train:** all timestamps **before** 2013-12-16 (includes validation week)
- **Test:** week **2013-12-16 → 2013-12-22** only

Uses **`best_hyperparams.json`** unless you pass `--no-best-params`.

**Models:**

| Model | Type | Training |
|-------|------|----------|
| `arima` | Statistical | `pmdarima.auto_arima`, seasonal period 144 (1 day) |
| `lstm` | Neural | 15 epochs, sequence length 144, MinMax scaling on train |
| `tcn` | Neural | Same setup, temporal convolution network |

**Neural training (LSTM & TCN):**
- Each **epoch**, the model sees all training sequences once
- **Terminal prints:** `Epoch 1/15  train_loss=0.00xxxx`
- **Saved artifacts** per model × square:
  - `data/outputs/forecast/training/training_lstm_square_4159.png` — loss curve
  - `.../training_lstm_square_4159.json` — numeric loss per epoch

**Forecast plots:**  
`data/outputs/forecast/arima_square_<id>.png` (9 total for 3 models × 3 squares)

**Metrics:** MAE, MAPE, RMSE in CSV + SQLite `ForecastRun` table

**Timing / hardware (for report tables):**
- `data/outputs/forecast/timing_table.csv`
- `data/outputs/forecast/timing_report.json` (includes `platform`, Python version, how times were measured)

**Predictions (for failure analysis):**
- `data/outputs/forecast/predictions/*.csv`

### 6. `run_failure_analysis` — Task 3 failure section

Finds the worst ~6-hour sliding window on the test week per model/square.

**Outputs in `data/outputs/failure_analysis/`:**
- `residuals_*_square_*.png` — full test week + residuals
- `worst_window_*_square_*.png` — zoom on worst interval
- `failure_report.json`, `failure_analysis.md` (draft text for your PDF)

---

## Viewing results

### Terminal
During `run_forecast`, watch epoch lines for LSTM/TCN.

### Files
Open PNG/CSV under `data/outputs/`.

### Browser
```bash
python manage.py runserver
```
- http://127.0.0.1:8000/ — gallery of all figures  
- http://127.0.0.1:8000/guide/ — this guide  

---

## What is an “epoch”?

One **epoch** = one full pass through the **training sequences** (sliding windows of 144 past values → next value). Loss should generally **decrease** as the network learns patterns. If loss flatlines or jumps, you may need more epochs, different learning rate, or more data (run full ingest, not `--max-files 2`).

ARIMA has no epochs; it fits statistical parameters in one search (`auto_arima`).

---

## Quick test vs full run

```bash
python manage.py ingest_raw --max-files 2
python manage.py build_series
python manage.py run_eda
python manage.py run_forecast
```

Full dataset ingest takes much longer (~5 GB raw).

---

## Assignment deliverables mapping

| Deliverable | Where it comes from |
|-------------|---------------------|
| Task 1 memory report | `ingest_report.json`, your PDF discussion |
| Task 2 figures | `data/outputs/eda/` |
| Task 3 plots + tables | `data/outputs/forecast/`, `metrics_square_*.csv` |
| Training time stats | printed + `metrics_summary.json` + `ForecastRun` |
| Code + README | repo root |

---

## AI / integrity note

Document tool use in your PDF. Be ready to explain ingest chunking, why Dec 16–22 is held out, and how each model uses history length 144.
