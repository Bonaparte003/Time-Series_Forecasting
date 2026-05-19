# Hyperparameter Experiment Journal

Tuning reference square: **5161**
Validation window: **2013-12-09** to **2013-12-15**
Test week (held out during tuning): **2013-12-16** to **2013-12-22**

## Methodology

1. **Phase 1 (baseline):** Default parameters to establish reference validation error.
2. **Phase 2 (grid search):** Search combinations in `EXPERIMENT_GRIDS` (settings).
3. **Selection:** Best validation MAE per model; saved to `best_hyperparams.json`.
4. **Final evaluation:** Run `run_forecast` — trains on all data before test week using best params.

## Phase 1 results

```
experiment_id model     val_mae    val_rmse                                                                                                                                                                        params
   P1-ets-001   ets  326.681047  486.447826                                                                                            {"seasonal_periods": 144, "trend": null, "seasonal": "add", "damped_trend": false}
  P1-lstm-002  lstm 2035.635617 2436.262709 {"seq_len": 144, "epochs": 15, "batch_size": 64, "lr": 0.001, "hidden": 64, "layers": 2, "early_stopping": true, "patience": 3, "holdout_intervals": 144, "min_delta": 1e-05}
   P1-tcn-003   tcn 2645.919222 3003.859374            {"seq_len": 144, "epochs": 15, "batch_size": 64, "lr": 0.001, "channels": 32, "early_stopping": true, "patience": 3, "holdout_intervals": 144, "min_delta": 1e-05}
```

## Phase 2 — grid search summary

```
         val_mae              val_rmse           
             min       mean        min       mean
model                                            
ets     314.6855   686.3099   477.5186   893.8821
lstm   1175.3041  1837.9578  1423.6762  2080.0822
tcn    1263.1838  1434.1766  1482.3681  1682.3622
```

### Iterative reasoning (for your report)

- **ETS:** Best experiment `P2-ets-007` (MAE=314.6855) vs worst in grid MAE=1777.1919. Selected params: `{"seasonal_periods": 144, "trend": "add", "seasonal": "add", "damped_trend": true}`.
- **LSTM:** Best experiment `P2-lstm-012` (MAE=1175.3041) vs worst in grid MAE=8053.5141. Selected params: `{"seq_len": 72, "epochs": 10, "batch_size": 64, "lr": 0.001, "hidden": 64, "layers": 2, "early_stopping": true, "patience": 3, "holdout_intervals": 144, "min_delta": 1e-05}`.
- **TCN:** Best experiment `P2-tcn-036` (MAE=1263.1838) vs worst in grid MAE=2073.1536. Selected params: `{"seq_len": 144, "epochs": 10, "batch_size": 64, "lr": 0.001, "channels": 32, "early_stopping": true, "patience": 3, "holdout_intervals": 144, "min_delta": 1e-05}`.

## Selected hyperparameters (final)

```json
{
  "ets": {
    "params": {
      "seasonal_periods": 144,
      "trend": "add",
      "seasonal": "add",
      "damped_trend": true
    },
    "val_mae": 314.6854531870751,
    "val_rmse": 477.51860278195,
    "val_mape": 23.484799278793453,
    "selected_from_experiment": "P2-ets-007",
    "phase": 2
  },
  "lstm": {
    "params": {
      "seq_len": 72,
      "epochs": 10,
      "batch_size": 64,
      "lr": 0.001,
      "hidden": 64,
      "layers": 2,
      "early_stopping": true,
      "patience": 3,
      "holdout_intervals": 144,
      "min_delta": 1e-05
    },
    "val_mae": 1175.3040935860515,
    "val_rmse": 1518.771521083454,
    "val_mape": 143.60139584351197,
    "selected_from_experiment": "P2-lstm-012",
    "phase": 2
  },
  "tcn": {
    "params": {
      "seq_len": 144,
      "epochs": 10,
      "batch_size": 64,
      "lr": 0.001,
      "channels": 32,
      "early_stopping": true,
      "patience": 3,
      "holdout_intervals": 144,
      "min_delta": 1e-05
    },
    "val_mae": 1263.183793292779,
    "val_rmse": 1496.3038054505128,
    "val_mape": 198.8750672297942,
    "selected_from_experiment": "P2-tcn-036",
    "phase": 2
  }
}
```