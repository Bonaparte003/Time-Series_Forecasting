"""Read pipeline results from SQLite (Django ORM) for the presentation UI."""

from __future__ import annotations

import json
from pathlib import Path

from django.conf import settings
from django.db.models import Count, Min

from traffic.models import ExperimentRun, ForecastRun, IngestionRun


def _db_path() -> str:
    return str(settings.DATABASES["default"]["NAME"])


def load_db_snapshot() -> dict:
    """Aggregate ORM data for server-rendered sections and JSON APIs."""
    ingestion = IngestionRun.objects.order_by("-started_at").first()
    ingestion_row = None
    if ingestion:
        ingestion_row = {
            "id": ingestion.pk,
            "started_at": ingestion.started_at.isoformat(),
            "files_processed": ingestion.files_processed,
            "rows_written": ingestion.rows_written,
            "memory_before_mb": ingestion.memory_before_mb,
            "memory_after_mb": ingestion.memory_after_mb,
        }

    forecast_rows = []
    for run in ForecastRun.objects.order_by("square_id", "model_name"):
        forecast_rows.append(
            {
                "square_id": run.square_id,
                "model": run.model_name.upper(),
                "model_key": run.model_name,
                "mae": round(run.mae, 1),
                "mape": round(run.mape, 1),
                "rmse": round(run.rmse, 1),
                "train_s": round(run.train_seconds, 2),
                "predict_s": round(run.predict_seconds, 3),
                "hyperparams": run.hyperparams,
                "created_at": run.created_at.isoformat(),
            }
        )

    metrics_by_square: dict[int, list] = {}
    for row in forecast_rows:
        sq = row["square_id"]
        metrics_by_square.setdefault(sq, []).append(row)
    for sq, rows in metrics_by_square.items():
        best = min(r["mae"] for r in rows)
        for r in rows:
            r["best"] = r["mae"] == best

    experiment_count = ExperimentRun.objects.count()
    best_per_model = (
        ExperimentRun.objects.values("model_name")
        .annotate(best_val_mae=Min("val_mae"))
        .order_by("model_name")
    )

    phases = (
        ExperimentRun.objects.values("phase", "phase_name")
        .annotate(n=Count("id"))
        .order_by("phase")
    )

    return {
        "db_path": _db_path(),
        "counts": {
            "ingestion": IngestionRun.objects.count(),
            "forecast": ForecastRun.objects.count(),
            "experiment": experiment_count,
        },
        "ingestion": ingestion_row,
        "forecast_rows": forecast_rows,
        "metrics_by_square": metrics_by_square,
        "best_per_model": list(best_per_model),
        "phases": [
            {"phase": p["phase"], "name": p["phase_name"], "count": p["n"]}
            for p in phases
        ],
        "has_data": experiment_count > 0 or len(forecast_rows) > 0,
    }


def serialize_experiments(
    *,
    model: str | None = None,
    phase: int | None = None,
    limit: int = 50,
) -> list[dict]:
    qs = ExperimentRun.objects.all().order_by("val_mae")
    if model:
        qs = qs.filter(model_name=model.lower())
    if phase is not None:
        qs = qs.filter(phase=phase)
    rows = []
    for run in qs[:limit]:
        rows.append(
            {
                "id": run.experiment_id,
                "phase": run.phase,
                "phase_name": run.phase_name,
                "model": run.model_name,
                "square_id": run.square_id,
                "val_mae": round(run.val_mae, 1),
                "val_mape": round(run.val_mape, 1),
                "val_rmse": round(run.val_rmse, 1),
                "train_s": round(run.train_seconds, 2),
                "params": run.params,
                "reasoning": run.reasoning[:120]
                + ("…" if len(run.reasoning) > 120 else ""),
            }
        )
    return rows


def orm_snippets() -> list[dict]:
    return [
        {
            "title": "Latest ingest run",
            "code": (
                "from traffic.models import IngestionRun\n"
                "run = IngestionRun.objects.order_by('-started_at').first()\n"
                "run.memory_before_mb, run.memory_after_mb"
            ),
        },
        {
            "title": "Test-week metrics (all squares)",
            "code": (
                "from traffic.models import ForecastRun\n"
                "for r in ForecastRun.objects.order_by('square_id', 'mae'):\n"
                "    print(r.square_id, r.model_name, r.mae, r.mape)"
            ),
        },
        {
            "title": "Best validation MAE per model",
            "code": (
                "from django.db.models import Min\n"
                "from traffic.models import ExperimentRun\n"
                "ExperimentRun.objects.values('model_name')\n"
                "    .annotate(best=Min('val_mae'))"
            ),
        },
        {
            "title": "Raw SQL (sqlite3 CLI)",
            "code": (
                f"sqlite3 {Path(_db_path()).name}\n"
                "SELECT square_id, model_name, mae, mape\n"
                "FROM traffic_forecastrun ORDER BY mae;"
            ),
        },
    ]
