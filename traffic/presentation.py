"""Context builder for the Formative 1 presentation UI."""

from __future__ import annotations

import json
from pathlib import Path

from django.conf import settings
from django.urls import reverse

from traffic.db_insights import load_db_snapshot, orm_snippets, serialize_experiments


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


def _read_snippet(rel_path: str, start: int, end: int) -> dict:
    path = Path(settings.BASE_DIR) / rel_path
    if not path.is_file():
        return {"file": rel_path, "lines": f"{start}-{end}", "code": "(file not found)"}
    lines = path.read_text(encoding="utf-8").splitlines()
    chunk = lines[start - 1 : end]
    return {
        "file": rel_path,
        "lines": f"{start}-{end}",
        "code": "\n".join(chunk),
    }


def build_presentation_context() -> dict:
    base = Path(settings.BASE_DIR)
    processed = Path(settings.PROCESSED_DIR)
    output = Path(settings.OUTPUT_DIR)

    ingest = _read_json(processed / "ingest_report.json") or {}
    metadata = _read_json(processed / "metadata.json") or {}
    metrics = _read_json(output / "forecast" / "metrics_summary.json") or {}
    hardware = _read_json(output / "forecast" / "hardware_environment.json") or {}

    squares = metadata.get("target_square_ids", [5161, 4159, 4556])
    top_id = metadata.get("top_traffic_square_id", 5161)

    eda_figures = [
        {
            "id": "pdf",
            "title": "Traffic PDF",
            "file": "eda/01_traffic_pdf.png",
            "talk": "Heavy right tail: few hotspots dominate; most grid cells are low-traffic.",
            **_figure("eda/01_traffic_pdf.png", "eda/01_traffic_pdf.png"),
        },
        {
            "id": "two_weeks",
            "title": "Three areas — two weeks",
            "file": "eda/02_three_areas_two_weeks.png",
            "talk": f"Compare square {top_id} (highest total), 4159, 4556 — daily seasonality vs baseline level.",
            **_figure("eda/02_three_areas_two_weeks.png", "eda/02_three_areas_two_weeks.png"),
        },
        {
            "id": "stationarity",
            "title": "Stationarity",
            "file": "eda/03_stationarity_square_5161.png",
            "talk": "ADF p≈0 on 5161 → reject unit root; still discuss rolling mean drift around holidays.",
            **_figure("eda/03_stationarity_square_5161.png", "eda/03_stationarity_square_5161.png"),
        },
        {
            "id": "decomposition",
            "title": "Decomposition",
            "file": "eda/04_decomposition_square_5161.png",
            "talk": "Strong 144-interval (24h) seasonality; residual spikes on event days.",
            **_figure("eda/04_decomposition_square_5161.png", "eda/04_decomposition_square_5161.png"),
        },
        {
            "id": "acf",
            "title": "ACF & PACF",
            "file": "eda/05_acf_pacf_square_5161.png",
            "talk": "Peaks at lag 144 and harmonics → daily periodicity drives model choice.",
            **_figure("eda/05_acf_pacf_square_5161.png", "eda/05_acf_pacf_square_5161.png"),
        },
        {
            "id": "heatmap",
            "title": "Spatial heatmap",
            "file": "eda/06_spatial_heatmap.png",
            "talk": "Traffic concentrates in city centre / corridors; edge cells are sparse.",
            **_figure("eda/06_spatial_heatmap.png", "eda/06_spatial_heatmap.png"),
        },
    ]

    models = ["ets", "lstm", "tcn"]
    model_labels = {"ets": "ETS (Holt–Winters)", "lstm": "LSTM", "tcn": "TCN"}
    forecast_plots = []
    for sq in squares:
        for model in models:
            rel = f"forecast/{model}_square_{sq}.png"
            forecast_plots.append(
                {
                    "square_id": sq,
                    "model": model,
                    "label": model_labels[model],
                    "title": f"{model_labels[model]} — {sq}",
                    **_figure(rel, f"forecast/{model}_square_{sq}.png"),
                }
            )

    db = load_db_snapshot()
    metrics_tables = []
    if db["metrics_by_square"]:
        for sq in sorted(db["metrics_by_square"]):
            metrics_tables.append(
                {"square_id": sq, "rows": db["metrics_by_square"][sq], "source": "sqlite"}
            )
    else:
        for sq in squares:
            rows = metrics.get(str(sq), [])
            if rows:
                best_mae = min(r["mae"] for r in rows)
                table_rows = []
                for r in rows:
                    table_rows.append(
                        {
                            "model": r["model"].upper(),
                            "mae": round(r["mae"], 1),
                            "mape": round(r["mape"], 1),
                            "rmse": round(r["rmse"], 1),
                            "train_s": round(r["train_seconds"], 2),
                            "predict_s": round(r["predict_seconds"], 3),
                            "best": r["mae"] == best_mae,
                        }
                    )
                metrics_tables.append(
                    {"square_id": sq, "rows": table_rows, "source": "json"}
                )

    ingest_db = db.get("ingestion")
    if ingest_db:
        ingest = {
            "memory_before_mb": ingest_db["memory_before_mb"],
            "memory_after_mb": ingest_db["memory_after_mb"],
            "files_processed": ingest_db["files_processed"],
            "rows_written": ingest_db["rows_written"],
            "from_db": True,
            "run_id": ingest_db["id"],
        }

    failure_samples = [
        {
            "model": "ETS",
            "square": 5161,
            "window": "2013-12-16 12:50 → 18:40",
            "note": "Largest errors at test-week start; multi-step drift from train end.",
            **_figure(
                "failure_analysis/worst_window_ets_square_5161.png",
                "failure/residuals_ets_square_5161.png",
            ),
        },
        {
            "model": "TCN",
            "square": 5161,
            "window": "2013-12-22 12:30 → 18:20",
            "note": "Pre-holiday spike; one-step still underestimates abrupt uplift.",
            **_figure(
                "failure_analysis/worst_window_tcn_square_5161.png",
                "failure/worst_window_tcn_square_5161.png",
            ),
        },
    ]

    snippets = [
        _read_snippet("traffic/services/loader.py", 36, 42),
        _read_snippet("traffic/services/forecast/training.py", 70, 94),
        _read_snippet("time_series_forecasting/settings.py", 55, 65),
    ]

    pipeline = [
        {
            "cmd": "python manage.py ingest_raw",
            "task": "Task 1",
            "file": "traffic/management/commands/ingest_raw.py → traffic/services/loader.py",
        },
        {
            "cmd": "python manage.py build_series",
            "task": "Prep",
            "file": "traffic/services/etl.py",
        },
        {
            "cmd": "python manage.py run_eda",
            "task": "Task 2",
            "file": "traffic/services/eda.py → data/outputs/eda/",
        },
        {
            "cmd": "python manage.py run_experiments",
            "task": "Tuning",
            "file": "traffic/services/experiments.py",
        },
        {
            "cmd": "python manage.py run_forecast",
            "task": "Task 3",
            "file": "traffic/services/forecast_runner.py",
        },
        {
            "cmd": "python manage.py run_failure_analysis",
            "task": "Task 3 · VIII",
            "file": "traffic/services/failure_analysis.py",
        },
    ]

    return {
        "title": "Formative 1 — Milan Mobile Traffic",
        "subtitle": "Comparative Time Series Analysis & Forecasting",
        "top_square_id": top_id,
        "squares": squares,
        "ingest": ingest,
        "hardware": hardware,
        "db": db,
        "orm_snippets": orm_snippets(),
        "experiments_preview": serialize_experiments(limit=12)
        if db["counts"]["experiment"]
        else [],
        "metrics_tables": metrics_tables,
        "eda_figures": eda_figures,
        "forecast_plots": forecast_plots,
        "failure_samples": failure_samples,
        "snippets": snippets,
        "pipeline": pipeline,
        "splits": [
            {"name": "Train (tuning)", "range": "Before 2013-12-09", "use": "run_experiments fit"},
            {"name": "Validation", "range": "2013-12-09 → 2013-12-15", "use": "Hyperparameter selection (square 5161)"},
            {"name": "Train (final)", "range": "Before 2013-12-16", "use": "run_forecast fit"},
            {"name": "Test", "range": "2013-12-16 → 2013-12-22", "use": "MAE / MAPE / RMSE & 9 plots"},
        ],
        "models_info": [
            {
                "id": "ets",
                "name": "ETS (Holt–Winters)",
                "type": "Statistical",
                "file": "traffic/services/forecast/ets.py",
                "inference": "Single forecast(1008) from train end — multi-step",
            },
            {
                "id": "lstm",
                "name": "LSTM",
                "type": "Neural (2-layer, seq_len=72)",
                "file": "traffic/services/forecast/lstm.py",
                "inference": "One-step-ahead with true test history",
            },
            {
                "id": "tcn",
                "name": "TCN",
                "type": "Neural (dilated conv, seq_len=144)",
                "file": "traffic/services/forecast/tcn.py",
                "inference": "One-step-ahead with true test history",
            },
        ],
        "artifact_paths": {
            "ingest_report": str(processed / "ingest_report.json"),
            "metadata": str(processed / "metadata.json"),
            "metrics_summary": str(output / "forecast" / "metrics_summary.json"),
            "timing": str(output / "forecast" / "timing_table.csv"),
            "failure_md": str(output / "failure_analysis" / "failure_analysis.md"),
            "experiment_journal": str(output / "experiments" / "experiment_journal.md"),
        },
    }
