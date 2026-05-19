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
- Worst window: **2013-12-17 10:10:00+00:00** → **2013-12-17 16:00:00+00:00**
- Window MAE: **240.6107**
- Peak absolute error: **306.5395**
- See: `worst_window_lstm_square_4159.png`

### LSTM — square 4556
- Worst window: **2013-12-22 02:30:00+00:00** → **2013-12-22 08:20:00+00:00**
- Window MAE: **317.0154**
- Peak absolute error: **428.9729**
- See: `worst_window_lstm_square_4556.png`

### LSTM — square 5161
- Worst window: **2013-12-21 00:30:00+00:00** → **2013-12-21 06:20:00+00:00**
- Window MAE: **2351.6629**
- Peak absolute error: **2953.0899**
- See: `worst_window_lstm_square_5161.png`

### TCN — square 4159
- Worst window: **2013-12-16 09:40:00+00:00** → **2013-12-16 15:30:00+00:00**
- Window MAE: **268.7710**
- Peak absolute error: **337.1617**
- See: `worst_window_tcn_square_4159.png`

### TCN — square 4556
- Worst window: **2013-12-22 02:30:00+00:00** → **2013-12-22 08:20:00+00:00**
- Window MAE: **444.2072**
- Peak absolute error: **497.4966**
- See: `worst_window_tcn_square_4556.png`

### TCN — square 5161
- Worst window: **2013-12-21 11:30:00+00:00** → **2013-12-21 17:20:00+00:00**
- Window MAE: **7916.3465**
- Peak absolute error: **9052.9819**
- See: `worst_window_tcn_square_5161.png`

## Suggested focus for report

Largest sustained error: **TCN** on square **5161** during **2013-12-21 11:30:00+00:00** – **2013-12-21 17:20:00+00:00**.

Possible causes to discuss (link to Task 2):
- Pre-holiday traffic surge (mid-December).
- Model lag on sharp ramps (neural nets using recursive preds).
- Non-stationarity / weekend vs weekday pattern shift.
- Single-square idiosyncrasy vs city-wide events.