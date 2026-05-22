"""Context builder for the Formative 1 presentation UI."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from django.conf import settings
from django.urls import reverse


def _read_json(path: Path) -> dict | list | None:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _output_url(relative: str) -> str:
    return reverse("serve_output", kwargs={"filepath": relative})


def _figure(rel_output: str, rel_docs: str | None = None) -> dict:
    """Prefer live outputs; fall back to committed docs/images."""
    root = Path(settings.OUTPUT_DIR)
    out = root / rel_output
    if out.is_file():
        return {"url": _output_url(rel_output), "path": str(out), "source": "data/outputs"}
    if rel_docs:
        docs = Path(settings.BASE_DIR) / "docs" / "images" / rel_docs
        if docs.is_file():
            return {
                "url": reverse("serve_doc_image", kwargs={"filepath": rel_docs}),
                "path": str(docs),
                "source": "docs/images",
            }
    return {"url": "", "path": rel_output, "source": "missing"}


def _truncate_cell(value: str, max_len: int = 40) -> str:
    value = value.strip()
    if len(value) <= max_len:
        return value
    return value[: max_len - 1] + "…"


def _read_csv_preview(rel_output: str, *, max_rows: int = 6) -> dict:
    path = Path(settings.OUTPUT_DIR) / rel_output
    if not path.is_file():
        return {
            "url": "",
            "path": rel_output,
            "headers": [],
            "rows": [],
            "total_rows": 0,
            "missing": True,
        }
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        all_rows = list(reader)
    if not all_rows:
        return {
            "url": _output_url(rel_output),
            "path": rel_output,
            "headers": [],
            "rows": [],
            "total_rows": 0,
            "missing": False,
        }
    headers = all_rows[0]
    body = []
    for row in all_rows[1 : max_rows + 1]:
        body.append([_truncate_cell(c) for c in row[: len(headers)]])
    return {
        "url": _output_url(rel_output),
        "path": rel_output,
        "headers": headers,
        "rows": body,
        "total_rows": len(all_rows) - 1,
        "missing": False,
    }


def _output_rel_paths(pattern: str) -> list[str]:
    root = Path(settings.OUTPUT_DIR)
    return sorted(str(p.relative_to(root)).replace("\\", "/") for p in root.glob(pattern))


def _gallery_figure(rel: str) -> dict:
    docs = rel.replace("failure_analysis/", "failure/")
    return {
        "title": Path(rel).stem.replace("_", " "),
        "caption": rel,
        **_figure(rel, docs),
    }


def _gallery_figures(rels: list[str]) -> list[dict]:
    return [_gallery_figure(r) for r in rels]


def _gallery_csvs(rels: list[str]) -> list[dict]:
    return [_read_csv_preview(r) for r in rels]


def _read_txt_block(rel_output: str, *, max_lines: int = 5) -> dict:
    path = Path(settings.OUTPUT_DIR) / rel_output
    if not path.is_file():
        return {"path": rel_output, "lines": [], "missing": True}
    return {
        "path": rel_output,
        "url": _output_url(rel_output),
        "lines": path.read_text(encoding="utf-8").splitlines()[:max_lines],
        "missing": False,
    }


_MODEL_LABELS = {"ets": "ETS", "lstm": "LSTM", "tcn": "TCN"}

# Test-week MAE / MAPE when metrics_summary.json is not on disk (see README).
_FALLBACK_TEST_METRICS: dict[int, list[dict]] = {
    5161: [
        {"model": "ets", "mae": 310.1, "mape": 26.1},
        {"model": "lstm", "mae": 107.1, "mape": 10.8},
        {"model": "tcn", "mae": 90.1, "mape": 9.4},
    ],
    4159: [
        {"model": "ets", "mae": 64.8, "mape": 28.1},
        {"model": "lstm", "mae": 16.0, "mape": 7.4},
        {"model": "tcn", "mae": 15.7, "mape": 6.7},
    ],
    4556: [
        {"model": "ets", "mae": 176.1, "mape": 48.6},
        {"model": "lstm", "mae": 32.0, "mape": 7.9},
        {"model": "tcn", "mae": 28.1, "mape": 6.5},
    ],
}


def _load_test_week_metrics(squares: list) -> dict[int, list[dict]]:
    """Parse metrics_summary.json keyed by square id."""
    raw = _read_json(Path(settings.OUTPUT_DIR) / "forecast" / "metrics_summary.json")
    if not isinstance(raw, dict):
        return {}
    out: dict[int, list[dict]] = {}
    for sq in squares:
        rows = raw.get(str(sq)) or raw.get(sq)
        if isinstance(rows, list):
            out[int(sq)] = rows
    return out


def _build_conclusion(*, squares: list) -> dict:
    """Closing section: test-week winners (MAE) and interpretation."""
    metrics_by_square = _load_test_week_metrics(squares)
    model_order = ["ets", "lstm", "tcn"]

    by_square: list[dict] = []
    for sq in squares:
        rows = metrics_by_square.get(int(sq)) or _FALLBACK_TEST_METRICS.get(int(sq), [])
        ranked = sorted(rows, key=lambda r: float(r.get("mae", float("inf"))))
        winner_row = ranked[0] if ranked else None
        winner = winner_row["model"] if winner_row else "tcn"
        table_rows = []
        for model in model_order:
            match = next((r for r in rows if r.get("model") == model), None)
            table_rows.append(
                {
                    "model": model,
                    "label": _MODEL_LABELS[model],
                    "mae": round(float(match["mae"]), 1) if match else None,
                    "mape": round(float(match["mape"]), 1) if match else None,
                    "is_winner": model == winner,
                }
            )
        by_square.append(
            {
                "square_id": sq,
                "winner": winner,
                "winner_label": _MODEL_LABELS.get(winner, winner.upper()),
                "rows": table_rows,
            }
        )

    winners = {b["square_id"]: b["winner"] for b in by_square}
    all_tcn = winners and all(w == "tcn" for w in winners.values())
    unique_winners = sorted(set(winners.values()))

    if all_tcn:
        verdict = (
            "TCN achieved the lowest test-week MAE on every target square "
            "(16–22 Dec 2013). LSTM was consistently second; ETS trailed, "
            "especially on the highest-traffic cell 5161."
        )
    elif len(unique_winners) == 1:
        label = _MODEL_LABELS.get(unique_winners[0], unique_winners[0])
        verdict = f"{label} achieved the lowest test-week MAE on all evaluated squares."
    else:
        verdict = (
            "The best model depends on the grid cell—spatial heterogeneity "
            "from EDA carries through to forecast accuracy."
        )

    return {
        "headline": "Conclusion — test week results",
        "verdict": verdict,
        "metric_note": "Winner selected by MAE on held-out week 2013-12-16 → 2013-12-22.",
        "paragraphs": [
            (
                "Hyperparameters were tuned on square 5161 using the validation week "
                "(9–15 Dec), then all three models were refit and scored on the test week. "
                "Neural models use one-step-ahead inference with true recent history; "
                "ETS issues a single 1,008-step forecast with explicit daily seasonality "
                "(period 144)."
            ),
            (
                "TCN’s full-day context (sequence length 144) helps capture the strong "
                "diurnal cycle seen in decomposition and ACF plots. That matters most where "
                "traffic is volatile—square 5161 (city hotspot) and mid-December ramps noted "
                "in failure analysis. ETS remains a fast, interpretable baseline but struggles "
                "when level shifts and holiday-week surges depart from the training pattern."
            ),
        ],
        "by_square": by_square,
        "reasons": [
            {
                "title": "Spatial heterogeneity",
                "text": (
                    "Hot squares (5161) have larger scale and sharper peaks; sequence "
                    "models adapt step-by-step, while ETS extrapolates one long seasonal path."
                ),
            },
            {
                "title": "Daily seasonality",
                "text": (
                    "EDA shows a 144-interval day/night rhythm. TCN and LSTM encode "
                    "recent days in their windows; TCN’s longer context (144 vs 72) helped "
                    "on the busiest square."
                ),
            },
            {
                "title": "December test week",
                "text": (
                    "Holiday-period level changes and short surges hurt ETS most. "
                    "Failure analysis links worst windows to pre-holiday ramps—NNs "
                    "tracked local dynamics better than a fixed seasonal extrapolation."
                ),
            },
            {
                "title": "Practical trade-off",
                "text": (
                    "ETS trains in seconds; TCN balances accuracy and cost (~7 s train "
                    "on 5161 vs ~25 s for LSTM). For production you might pick TCN where "
                    "MAE matters and ETS where speed and simplicity dominate."
                ),
            },
        ],
        "figure": {
            "title": "TCN forecast — square 5161 (test week)",
            "caption": "forecast/tcn_square_5161.png",
            **_figure("forecast/tcn_square_5161.png", "forecast/tcn_square_5161.png"),
        },
    }


def _build_introduction(*, squares: list) -> dict:
    """Hero section at the top of the presentation scroll deck."""
    sq_label = ", ".join(str(s) for s in squares)
    return {
        "headline": "Milan mobile network traffic forecasting",
        "tagline": "Comparative time series analysis on the TIM Milan dataset (Barlacchi et al., 2015).",
        "paragraphs": [
            (
                "We study anonymized mobile internet activity (CDR proxy) across roughly "
                "10,000 grid cells covering the Milan metropolitan area. Raw tab-separated "
                "TIM files are ingested in chunks, aggregated to 10-minute intervals, and "
                "stored as columnar Parquet for efficient analysis."
            ),
            (
                "Exploratory analysis highlights strong spatial heterogeneity—few hot "
                "squares dominate total traffic—and a clear daily rhythm (144 intervals per "
                "day). We compare three forecasters on representative areas: Holt–Winters ETS, "
                "LSTM, and TCN, tuned on a validation week and evaluated on a held-out test week."
            ),
        ],
        "highlights": [
            {
                "label": "Target squares",
                "value": f"{sq_label} (5161 = highest traffic)",
            },
            {
                "label": "Timeline",
                "value": "Nov–Dec 2013 · val 9–15 Dec · test 16–22 Dec",
            },
            {
                "label": "Models",
                "value": "ETS · LSTM · TCN (daily seasonality / seq length 72–144)",
            },
            {
                "label": "Pipeline",
                "value": "ingest → series → EDA → experiments → forecast → failure analysis",
            },
        ],
        "dataset_label": "TIM Milan on Harvard Dataverse",
        "dataset_url": (
            "https://dataverse.harvard.edu/dataset.xhtml"
            "?persistentId=doi:10.7910/DVN/EGZHFV"
        ),
        "figure": {
            "title": "Spatial heatmap — target squares marked with ★",
            "caption": "eda/06_spatial_heatmap.png",
            **_figure("eda/06_spatial_heatmap.png", "eda/06_spatial_heatmap.png"),
        },
    }


def _build_overview_journey(*, ingest: dict, squares: list) -> list[dict]:
    """Single scroll deck: every PNG and CSV under data/outputs."""
    models = ["ets", "lstm", "tcn"]
    forecast_plots = [f"forecast/{m}_square_{sq}.png" for sq in squares for m in models]
    training_plots = _output_rel_paths("forecast/training/*.png")
    failure_residuals = _output_rel_paths("failure_analysis/residuals_*.png")
    failure_worst = _output_rel_paths("failure_analysis/worst_window_*.png")
    prediction_csvs = _output_rel_paths("forecast/predictions/*.csv")
    metrics_csvs = [f"forecast/metrics_square_{sq}.csv" for sq in squares]

    mem_before = ingest.get("memory_before_mb")
    mem_after = ingest.get("memory_after_mb")
    kicker = ""
    if mem_before and mem_after:
        kicker = (
            f"{ingest.get('files_processed', 62)} files · "
            f"{ingest.get('rows_written', 0):,} rows · "
            f"RAM {float(mem_before):.1f} → {float(mem_after):.1f} MB"
        )

    return [
        {
            "id": "ingest",
            "step": 1,
            "phase": "Ingest",
            "title": "Raw TIM → daily Parquet",
            "cmd": "python manage.py ingest_raw · build_series",
            "kicker": kicker,
            "figure_groups": [],
            "csvs": [],
            "texts": [],
        },
        {
            "id": "eda",
            "step": 2,
            "phase": "EDA",
            "title": "Exploratory analysis",
            "cmd": "python manage.py run_eda",
            "kicker": "",
            "figure_groups": [
                {
                    "label": "Plots",
                    "figures": _gallery_figures(
                        [
                            "eda/01_traffic_pdf.png",
                            "eda/02_three_areas_two_weeks.png",
                            "eda/03_stationarity_square_5161.png",
                            "eda/03_stationarity_square_4159.png",
                            "eda/03_stationarity_square_4556.png",
                            "eda/04_decomposition_square_5161.png",
                            "eda/05_acf_pacf_square_5161.png",
                            "eda/06_spatial_heatmap.png",
                        ]
                    ),
                },
            ],
            "csvs": _gallery_csvs(["eda/07_outlier_sample.csv"]),
            "texts": [_read_txt_block(f"eda/03_adf_square_{sq}.txt") for sq in squares],
        },
        {
            "id": "experiments",
            "step": 3,
            "phase": "Tuning",
            "title": "Validation-week experiments",
            "cmd": "python manage.py run_experiments",
            "kicker": "Square 5161",
            "figure_groups": [
                {"label": "Training curves", "figures": _gallery_figures(training_plots)},
            ],
            "csvs": _gallery_csvs(["experiments/experiments_log.csv"]),
            "texts": [],
        },
        {
            "id": "forecast",
            "step": 4,
            "phase": "Forecast",
            "title": "Test week — ETS · LSTM · TCN",
            "cmd": "python manage.py run_forecast",
            "kicker": "16–22 Dec 2013 · squares 5161, 4159, 4556",
            "figure_groups": [
                {"label": "Forecast plots", "figures": _gallery_figures(forecast_plots)},
            ],
            "csvs": _gallery_csvs(
                metrics_csvs + ["forecast/timing_table.csv"] + prediction_csvs
            ),
            "texts": [],
        },
        {
            "id": "failure",
            "step": 5,
            "phase": "Failure analysis",
            "title": "Residuals & worst windows",
            "cmd": "python manage.py run_failure_analysis",
            "kicker": f"{len(failure_residuals) + len(failure_worst)} plots",
            "figure_groups": [
                {"label": "Residuals", "figures": _gallery_figures(failure_residuals)},
                {"label": "Worst windows", "figures": _gallery_figures(failure_worst)},
            ],
            "csvs": [],
            "texts": [],
        },
    ]


def build_presentation_context() -> dict:
    processed = Path(settings.PROCESSED_DIR)
    ingest = _read_json(processed / "ingest_report.json") or {}
    metadata = _read_json(processed / "metadata.json") or {}
    squares = metadata.get("target_square_ids", [5161, 4159, 4556])

    return {
        "title": "Formative 1 — Milan Mobile Traffic",
        "subtitle": "Comparative Time Series Analysis & Forecasting",
        "squares": squares,
        "introduction": _build_introduction(squares=squares),
        "conclusion": _build_conclusion(squares=squares),
        "overview_journey": _build_overview_journey(ingest=ingest, squares=squares),
    }
