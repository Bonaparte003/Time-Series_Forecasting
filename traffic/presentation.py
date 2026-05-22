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
        "overview_journey": _build_overview_journey(ingest=ingest, squares=squares),
    }
