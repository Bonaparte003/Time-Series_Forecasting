# Forecasting methods reference

Technical description of every model, preprocessing step, training procedure, and evaluation protocol used in **Task 3**. Aligned with the formative brief (*one-step-ahead prediction*) and the implementation under `traffic/services/forecast/`.

**See also:** [Directory reference](DIRECTORY_REFERENCE.md) · [Project guide](../PROJECT_GUIDE.md) · [README](../README.md)

---

## Problem formulation

- **Observations:** internet traffic proxy (CDR count) per square \(a\), every **10 minutes**.
- **Seasonal cycle:** 144 intervals = 24 hours (`SEASONAL_PERIOD` in settings).
- **Assignment target:** at time \(t\), use history \(x_{t-\text{seq\_len}+1}, \ldots, x_t\) and predict \(x_{t+1}\).
- **Final evaluation week:** 2013-12-16 → 2013-12-22 (**1,008** intervals), never used in training or hyperparameter tuning.

Each of the three target squares (**5161**, **4159**, **4556**) is modeled **independently** with the same three algorithms.

---

## Data splits

| Phase | Date range | Role |
|-------|------------|------|
| Train (tuning) | Before 2013-12-09 | Fit models in `run_experiments` |
| Validation | 2013-12-09 → 2013-12-15 | Select hyperparameters (square 5161 only) |
| Train (final) | Before 2013-12-16 | Fit models in `run_forecast` (includes validation week) |
| Test | 2013-12-16 → 2013-12-22 | Report MAE, MAPE, RMSE and overlay plots |

Implemented in `BaseForecaster.split_train_test()` and `split_for_tuning()` (`traffic/services/forecast/base.py`).

---

## Shared neural preprocessing

| Step | Detail | Code |
|------|--------|------|
| Scaling | `sklearn.preprocessing.MinMaxScaler` fit on **train** values only | `BaseForecaster.scaler` |
| Sequences | Windows of length `seq_len`; target = next scaled value | `make_sequences()` |
| Loss | MSE on scaled targets | `train_epochs()` in `training.py` |
| Optimizer | Adam, learning rate from hyperparams (default `1e-3`) | `training.py` |
| Early stopping | Last `holdout_intervals` (default **144**) **training** sequences held out; restore best val MSE weights; patience **3** | `build_sequence_loaders()`, `train_epochs()` |
| Device | CUDA if available, else CPU | `get_torch_device()` |

**Important:** Early-stopping validation uses **one-step** windows from the **training** series tail, not the Dec 9–15 validation week. Hyperparameter **selection** uses the separate validation week via `run_experiments`.

---

## Model 1 — ETS (Holt–Winters)

**Type:** Classical statistical baseline (required: one non-neural model).

| Item | Choice |
|------|--------|
| Library | `statsmodels.tsa.holtwinters.ExponentialSmoothing` |
| Seasonality | Additive, period **144** |
| Trend | Tuned: `None` or `"add"` |
| Damped trend | Tuned when trend is additive |
| Input | Full training series (univariate, no MinMax) |

**Training:** Single fit on all pre-test data (`EtsForecaster._fit`).

**Test inference:** One call `fitted.forecast(n_test)` for the full test week (**1,008** steps) from the state at the end of training. This is a **direct multi-step** forecast from the training origin, not stepwise oracle history like the neural nets.

**Best hyperparameters (from grid on square 5161):** `trend="add"`, `seasonal="add"`, `damped_trend=true`, `seasonal_periods=144`.

**Code:** `traffic/services/forecast/ets.py`

---

## Model 2 — LSTM

**Type:** Recurrent neural network.

| Item | Choice |
|------|--------|
| Architecture | 2-layer `nn.LSTM` (hidden size tuned, default 64) → `nn.Linear` → 1 output |
| Input shape | `(batch, seq_len, 1)` |
| Default / tuned `seq_len` | **72** (12 h) in `best_hyperparams.json`; grid also tried 144 |
| Epochs | Tuned (10 in best config); max 15 in defaults |

**Training:** Shuffle batches over training sequences; optional early stopping on train-tail holdout.

**Test inference — one-step-ahead (assignment-compliant):**

For each test index \(i = 0, \ldots, 1007\):

1. Build window = last `seq_len` **true** scaled values: `train` concatenated with `test[0:i]`.
2. Forward pass → \(\hat{x}\) at `test[i]`.
3. Append **true** scaled `test[i]` to history (not the prediction).

Implemented in `one_step_ahead_predict()` (`traffic/services/forecast/training.py`), called from `LstmForecaster.fit_predict()`.

**Code:** `traffic/services/forecast/lstm.py`, `LSTMNet`
-
---

## Model 3 — TCN (Temporal Convolutional Network)

**Type:** Convolutional neural network with dilated causal-style blocks.

| Item | Choice |
|------|--------|
| Blocks | 3× `TCNBlock` with dilations 1, 2, 4; residual connections |
| Channels | Tuned (32 in best config) |
| Head | Linear on last time step of conv output |
| Tuned `seq_len` | **144** (24 h) in `best_hyperparams.json` |

**Training / inference:** Same pipeline as LSTM (`lstm_targets=False` in loaders for target tensor shape).

**Code:** `traffic/services/forecast/tcn.py`, `TCNNet`

---

## Evaluation metrics

Computed on the test week in original traffic units (`traffic/services/forecast/metrics.py`):

| Metric | Formula |
|--------|---------|
| **MAE** | \(\mathrm{mean}(|y - \hat{y}|)\) |
| **RMSE** | \(\sqrt{\mathrm{mean}((y - \hat{y})^2)}\) |
| **MAPE** | \(\mathrm{mean}(|y - \hat{y}| / \max(|y|, \epsilon)) \times 100\) |

Timing: `time.perf_counter()` around fit and predict phases; hardware snapshot in `hardware_environment.json`.

---

## Hyperparameter tuning (`run_experiments`)

| Phase | Runs | Description |
|-------|-----:|-------------|
| 1 — Baseline | 3 | Default params, one run per model on square **5161** |
| 2 — Grid | 36 | Cartesian product of grids in `settings.EXPERIMENT_GRIDS` |

**Selection criterion:** lowest validation-week **MAE** on square 5161. Winners written to `data/processed/best_hyperparams.json`.

**Note:** Validation-week evaluation uses the same `fit_predict()` path as final forecast (including one-step-ahead for NNs). Tuning MAE on 5161 can still be high for early grid configs; the selected LSTM/TCN params are the best **within the grid**, not necessarily globally optimal.

---

## Comparing models fairly in the report

| Aspect | ETS | LSTM / TCN |
|--------|-----|------------|
| Test protocol | Single 1008-step forecast from train end | 1008 one-step forecasts with **true** test history in each window |
| Seasonality | Explicit daily period 144 | Implicit via lag window length |
| Strengths | Stable on strong daily pattern (5161) | Strong when local dynamics match train scale (4159, 4556 after fix) |
| Discuss | Structural difference in §VII | Training curves show scaled one-step fit; test metrics show oracle one-step |

---

## Current test-week results (after one-step-ahead fix)

From `data/outputs/forecast/metrics_summary.json` (re-run `run_forecast` to refresh).

### Square 5161

| Model | MAE | MAPE % | RMSE |
|-------|----:|-------:|-----:|
| ETS | 310.1 | 26.1 | 470.0 |
| **TCN** | **90.1** | **9.4** | **130.3** |
| LSTM | 107.1 | 10.8 | 159.3 |

### Square 4159

| Model | MAE | MAPE % | RMSE |
|-------|----:|-------:|-----:|
| **TCN** | **15.7** | **6.7** | **21.3** |
| LSTM | 16.0 | 7.4 | 21.3 |
| ETS | 64.8 | 28.1 | 85.7 |

### Square 4556

| Model | MAE | MAPE % | RMSE |
|-------|----:|-------:|-----:|
| **TCN** | **28.1** | **6.5** | **37.7** |
| LSTM | 32.0 | 7.9 | 41.0 |
| ETS | 176.1 | 48.6 | 196.0 |

**Interpretation:** Neural nets with one-step oracle history outperform ETS on 4159/4556; on 5161 TCN/LSTM beat ETS on MAE but ETS remains competitive on MAPE. Best model is **square-dependent** — tie to Task 2 (heterogeneity, seasonality, outliers).

---

## Failure analysis (`run_failure_analysis`)

- **Sliding window:** 36 intervals = 6 hours.
- **Worst window:** highest mean absolute error per model × square.
- **Artifacts:** residual time series plots, worst-window overlays, auto-draft `failure_analysis.md`.

---

## Key source files

| File | Responsibility |
|------|----------------|
| `forecast/base.py` | Splits, sequences, `ForecastResult` |
| `forecast/ets.py` | Holt–Winters |
| `forecast/lstm.py` | LSTM training + predict |
| `forecast/tcn.py` | TCN training + predict |
| `forecast/training.py` | Training loop, early stopping, `one_step_ahead_predict` |
| `forecast/factory.py` | `create_forecaster(name, params)` |
| `forecast_runner.py` | Task 3 orchestration, plots, CSV export |
| `experiments.py` | Grid search, `best_hyperparams.json` |
