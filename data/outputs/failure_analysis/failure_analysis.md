# Failure Analysis (auto-generated)

Use this draft for **Task 3 Failure Analysis** in your report; add interpretation.

Sliding window: 36 × 10-min intervals.

## Per model / square

### ETS — square 4159
- Worst window: **2013-12-22 08:00:00+00:00** → **2013-12-22 13:50:00+00:00**
- Window MAE: **165.6193**
- Peak absolute error: **235.2111**
- See: `worst_window_ets_square_4159.png`

### ETS — square 4556
- Worst window: **2013-12-19 05:50:00+00:00** → **2013-12-19 11:40:00+00:00**
- Window MAE: **320.8055**
- Peak absolute error: **409.7378**
- See: `worst_window_ets_square_4556.png`

### ETS — square 5161
- Worst window: **2013-12-16 12:50:00+00:00** → **2013-12-16 18:40:00+00:00**
- Window MAE: **1135.4997**
- Peak absolute error: **1544.4751**
- See: `worst_window_ets_square_5161.png`

### LSTM — square 4159
- Worst window: **2013-12-18 08:50:00+00:00** → **2013-12-18 14:40:00+00:00**
- Window MAE: **31.7734**
- Peak absolute error: **122.0498**
- See: `worst_window_lstm_square_4159.png`

### LSTM — square 4556
- Worst window: **2013-12-17 17:50:00+00:00** → **2013-12-17 23:40:00+00:00**
- Window MAE: **49.3518**
- Peak absolute error: **258.8202**
- See: `worst_window_lstm_square_4556.png`

### LSTM — square 5161
- Worst window: **2013-12-17 13:20:00+00:00** → **2013-12-17 19:10:00+00:00**
- Window MAE: **255.4079**
- Peak absolute error: **604.7648**
- See: `worst_window_lstm_square_5161.png`

### TCN — square 4159
- Worst window: **2013-12-18 09:00:00+00:00** → **2013-12-18 14:50:00+00:00**
- Window MAE: **28.0009**
- Peak absolute error: **119.7900**
- See: `worst_window_tcn_square_4159.png`

### TCN — square 4556
- Worst window: **2013-12-18 15:30:00+00:00** → **2013-12-18 21:20:00+00:00**
- Window MAE: **45.8452**
- Peak absolute error: **302.1330**
- See: `worst_window_tcn_square_4556.png`

### TCN — square 5161
- Worst window: **2013-12-22 12:30:00+00:00** → **2013-12-22 18:20:00+00:00**
- Window MAE: **187.1552**
- Peak absolute error: **532.8788**
- See: `worst_window_tcn_square_5161.png`

## Suggested focus for report

Largest sustained error: **ETS** on square **5161** during **2013-12-16 12:50:00+00:00** – **2013-12-16 18:40:00+00:00**.

Possible causes to discuss (link to Task 2):
- Pre-holiday traffic surge (mid-December).
- Model lag on sharp ramps or holiday surges (one-step-ahead NNs vs ETS seasonality).
- Non-stationarity / weekend vs weekday pattern shift.
- Single-square idiosyncrasy vs city-wide events.