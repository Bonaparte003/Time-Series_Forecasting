"""Orchestrate Task 3 forecasting across models and target squares."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from django.conf import settings

from traffic.models import ForecastRun
from traffic.services.etl import load_metadata, load_square_series
from traffic.services.experiments import load_best_params
from traffic.services.failure_analysis import save_predictions
from traffic.services.forecast.factory import create_forecaster, default_params
from traffic.services.forecast.metrics import mae, mape, rmse
from traffic.services.hardware import collect_environment, save_environment
from traffic.services.verbose import LogFn


def _resolve_params(model_name: str, use_best: bool) -> dict:
    if use_best:
        best = load_best_params()
        if best and model_name in best:
            return best[model_name]["params"]
    return default_params(model_name)


def run_all_forecasts(
    models: tuple[str, ...] | None = None,
    *,
    log: LogFn = None,
    use_best_params: bool = True,
) -> dict:
    models = models or settings.FORECAST_MODELS
    metadata = load_metadata()
    square_ids = metadata["target_square_ids"]
    out_dir = Path(settings.OUTPUT_DIR) / "forecast"
    training_dir = out_dir / "training"
    out_dir.mkdir(parents=True, exist_ok=True)
    training_dir.mkdir(parents=True, exist_ok=True)
    fit_verbose = log is not None

    env_path = out_dir / "hardware_environment.json"
    save_environment(env_path)
    if log:
        log(
            f"Task 3 forecast — squares {square_ids} | test week "
            f"{settings.TEST_START} → {settings.TEST_END}"
        )
        log(f"Models: {', '.join(models)} | best_params={use_best_params}")

    all_metrics: dict[int, list[dict]] = {sid: [] for sid in square_ids}
    timing_rows: list[dict] = []

    for square_id in square_ids:
        if log:
            log(f"\n=== Square {square_id} ===")
        series = load_square_series(square_id)
        if log:
            log(f"  series length: {len(series):,} intervals")
        for model_name in models:
            params = _resolve_params(model_name, use_best_params)
            if log:
                log(f"\n--- Model: {model_name} ---")
                log(f"    Hyperparameters: {params}")
            forecaster = create_forecaster(model_name, params)
            train, test = forecaster.split_train_test(series)
            if log:
                log(f"    train: {len(train):,} | test: {len(test):,}")
            result = forecaster.fit_predict(
                train,
                test,
                verbose=fit_verbose,
                save_curve=True,
                curve_dir=training_dir,
                square_id=square_id,
            )

            save_predictions(
                model_name, square_id, result.index, result.y_true, result.y_pred
            )

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(result.index, result.y_true, label="Actual", linewidth=1)
            ax.plot(result.index, result.y_pred, label="Predicted", linewidth=1, alpha=0.85)
            ax.set_title(
                f"{model_name.upper()} — square {square_id} (Dec 16–22, 2013)"
            )
            ax.legend()
            ax.set_xlabel("Time")
            ax.set_ylabel("Internet traffic")
            fig.tight_layout()
            plot_path = out_dir / f"{model_name}_square_{square_id}.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            if log:
                log(f"    Saved forecast plot: {plot_path.name}")

            th = result.training_history or {}
            metrics = {
                "square_id": square_id,
                "model": model_name,
                "hyperparams": json.dumps(params),
                "mae": mae(result.y_true, result.y_pred),
                "mape": mape(result.y_true, result.y_pred),
                "rmse": rmse(result.y_true, result.y_pred),
                "train_seconds": result.train_seconds,
                "predict_seconds": result.predict_seconds,
                "final_train_loss": (
                    result.epoch_losses[-1] if result.epoch_losses else None
                ),
                "epochs_run": len(th["train_loss"]) if th.get("train_loss") else None,
                "best_epoch": th.get("best_epoch"),
                "stopped_early": th.get("stopped_early"),
                "best_val_loss": (
                    th["val_loss"][th["best_epoch"] - 1]
                    if th.get("val_loss") and th.get("best_epoch")
                    else None
                ),
                "best_val_mae": (
                    th["val_mae"][th["best_epoch"] - 1]
                    if th.get("val_mae") and th.get("best_epoch")
                    else None
                ),
            }
            all_metrics[square_id].append(metrics)
            timing_rows.append({**metrics, "total_seconds": metrics["train_seconds"] + metrics["predict_seconds"]})
            if log:
                train_note = ""
                if th:
                    train_note = (
                        f" | epochs={metrics['epochs_run']} "
                        f"best={metrics['best_epoch']}"
                        f"{' early_stop' if metrics['stopped_early'] else ''}"
                    )
                log(
                    f"    MAE={metrics['mae']:.4f} MAPE={metrics['mape']:.2f}% "
                    f"RMSE={metrics['rmse']:.4f} | "
                    f"train={metrics['train_seconds']:.1f}s predict={metrics['predict_seconds']:.1f}s"
                    f"{train_note}"
                )

            ForecastRun.objects.update_or_create(
                square_id=square_id,
                model_name=model_name,
                defaults={
                    "mae": metrics["mae"],
                    "mape": metrics["mape"],
                    "rmse": metrics["rmse"],
                    "train_seconds": metrics["train_seconds"],
                    "predict_seconds": metrics["predict_seconds"],
                    "hyperparams": params,
                },
            )

    for square_id, rows in all_metrics.items():
        table = pd.DataFrame(rows)
        path = out_dir / f"metrics_square_{square_id}.csv"
        table.to_csv(path, index=False)
        if log:
            log(f"Metrics table: {path.name}")

    timing_df = pd.DataFrame(timing_rows)
    timing_path = out_dir / "timing_table.csv"
    timing_df.to_csv(timing_path, index=False)

    timing_report = {
        "environment": collect_environment(),
        "measurement_note": (
            "train_seconds = model.fit (ETS / NN training); "
            "predict_seconds = one-step walk-forward over test week; "
            "recorded via time.perf_counter() on local machine."
        ),
        "rows": timing_rows,
    }
    timing_json_path = out_dir / "timing_report.json"
    with open(timing_json_path, "w", encoding="utf-8") as f:
        json.dump(timing_report, f, indent=2)

    summary_path = out_dir / "metrics_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    if log:
        log(f"Timing: {timing_path.name} | {timing_json_path.name}")

    return {
        "square_ids": square_ids,
        "models": list(models),
        "output_dir": str(out_dir),
        "training_dir": str(training_dir),
        "timing_table": str(timing_path),
        "timing_report": str(timing_json_path),
        "hardware": str(env_path),
    }
