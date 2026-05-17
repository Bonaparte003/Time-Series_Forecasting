"""
Hyperparameter experimentation with documented phases.

Phase 1: baseline defaults on validation week.
Phase 2: grid search on validation week (Dec 9–15).
Best params per model → best_hyperparams.json for final test evaluation.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pandas as pd
from django.conf import settings

from traffic.models import ExperimentRun
from traffic.services.etl import load_metadata, load_square_series
from traffic.services.forecast.factory import create_forecaster, default_params
from traffic.services.forecast.metrics import mae, mape, rmse
from traffic.services.verbose import LogFn


def _param_combinations(model_name: str, *, quick: bool = False) -> list[dict]:
    """All grid combinations merged with model defaults."""
    grids = settings.EXPERIMENT_GRIDS_QUICK if quick else settings.EXPERIMENT_GRIDS
    grid = grids.get(model_name, {})
    if not grid:
        return [default_params(model_name)]
    keys = list(grid.keys())
    combos = []
    for values in itertools.product(*(grid[k] for k in keys)):
        params = {**default_params(model_name), **dict(zip(keys, values))}
        combos.append(params)
    return combos


def _evaluate_on_validation(
    model_name: str,
    params: dict,
    series: pd.Series,
    *,
    fit_verbose: bool,
) -> dict:
    forecaster = create_forecaster(model_name, params)
    train, val = forecaster.split_for_tuning(series)
    result = forecaster.fit_predict(
        train,
        val,
        verbose=fit_verbose,
        save_training_curve=False,
    )
    return {
        "val_mae": mae(result.y_true, result.y_pred),
        "val_mape": mape(result.y_true, result.y_pred),
        "val_rmse": rmse(result.y_true, result.y_pred),
        "train_seconds": result.train_seconds,
        "predict_seconds": result.predict_seconds,
    }


def _reasoning_for_phase(phase_cfg: dict, model_name: str, params: dict) -> str:
    base = phase_cfg["reasoning"]
    if phase_cfg["use_grid"]:
        return f"{base} Model={model_name}; params={json.dumps(params, sort_keys=True)}."
    return f"{base} Model={model_name}; default params={json.dumps(params, sort_keys=True)}."


def run_experiments(
    *,
    square_id: int | None = None,
    models: tuple[str, ...] | None = None,
    log: LogFn = None,
    quick: bool = False,
) -> dict:
    models = models or settings.FORECAST_MODELS
    metadata = load_metadata()
    tune_square = square_id or metadata["top_traffic_square_id"]
    series = load_square_series(tune_square)
    fit_verbose = log is not None

    out_dir = Path(settings.EXPERIMENTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    experiment_counter = 0

    if log:
        log(
            f"Experiments on square {tune_square} | validation "
            f"{settings.VAL_START} → {settings.VAL_END} | quick={quick}"
        )

    for phase_cfg in settings.EXPERIMENT_PHASES:
        phase = phase_cfg["phase"]
        if log:
            log(f"\n######## Phase {phase}: {phase_cfg['name']} ########")

        for model_name in models:
            if phase_cfg["use_grid"]:
                param_list = _param_combinations(model_name, quick=quick)
            else:
                param_list = [default_params(model_name)]

            if log:
                log(f"  {model_name}: {len(param_list)} configuration(s)")

            for params in param_list:
                experiment_counter += 1
                exp_id = f"P{phase}-{model_name}-{experiment_counter:03d}"
                if log:
                    log(f"\n[{exp_id}] {model_name} {params}")

                metrics = _evaluate_on_validation(
                    model_name, params, series, fit_verbose=fit_verbose
                )
                if log:
                    log(
                        f"  val MAE={metrics['val_mae']:.4f} "
                        f"RMSE={metrics['val_rmse']:.4f} "
                        f"train={metrics['train_seconds']:.1f}s"
                    )
                reasoning = _reasoning_for_phase(phase_cfg, model_name, params)

                row = {
                    "experiment_id": exp_id,
                    "phase": phase,
                    "phase_name": phase_cfg["name"],
                    "model": model_name,
                    "square_id": tune_square,
                    "params": json.dumps(params),
                    "reasoning": reasoning,
                    **metrics,
                }
                rows.append(row)

                ExperimentRun.objects.update_or_create(
                    experiment_id=exp_id,
                    defaults={
                        "phase": phase,
                        "phase_name": phase_cfg["name"],
                        "model_name": model_name,
                        "square_id": tune_square,
                        "params": params,
                        "val_mae": metrics["val_mae"],
                        "val_mape": metrics["val_mape"],
                        "val_rmse": metrics["val_rmse"],
                        "train_seconds": metrics["train_seconds"],
                        "reasoning": reasoning,
                    },
                )

    df = pd.DataFrame(rows)
    csv_path = out_dir / "experiments_log.csv"
    df.to_csv(csv_path, index=False)

    best = _select_best_params(df)
    best_path = Path(settings.BEST_PARAMS_PATH)
    best_path.parent.mkdir(parents=True, exist_ok=True)
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    journal_path = out_dir / "experiment_journal.md"
    _write_journal(df, best, tune_square, journal_path)

    if log:
        log(f"\nWrote {csv_path.name} ({len(rows)} runs)")
        log(f"Best params: {best_path}")
        log(f"Journal: {journal_path}")

    return {
        "tuning_square_id": tune_square,
        "experiments_csv": str(csv_path),
        "best_params_path": str(best_path),
        "journal_path": str(journal_path),
        "n_experiments": len(rows),
    }


def _select_best_params(df: pd.DataFrame) -> dict:
    """Pick lowest validation MAE per model (phase 2 preferred if present)."""
    best: dict[str, dict] = {}
    for model in df["model"].unique():
        sub = df[df["model"] == model]
        phase2 = sub[sub["phase"] == 2]
        pick_from = phase2 if len(phase2) else sub
        row = pick_from.loc[pick_from["val_mae"].idxmin()]
        best[model] = {
            "params": json.loads(row["params"]),
            "val_mae": float(row["val_mae"]),
            "val_rmse": float(row["val_rmse"]),
            "val_mape": float(row["val_mape"]),
            "selected_from_experiment": row["experiment_id"],
            "phase": int(row["phase"]),
        }
    return best


def _write_journal(
    df: pd.DataFrame, best: dict, square_id: int, path: Path
) -> None:
    lines = [
        "# Hyperparameter Experiment Journal",
        "",
        f"Tuning reference square: **{square_id}**",
        f"Validation window: **{settings.VAL_START}** to **{settings.VAL_END}**",
        f"Test week (held out during tuning): **{settings.TEST_START}** to **{settings.TEST_END}**",
        "",
        "## Methodology",
        "",
        "1. **Phase 1 (baseline):** Default parameters to establish reference validation error.",
        "2. **Phase 2 (grid search):** Search combinations in `EXPERIMENT_GRIDS` (settings).",
        "3. **Selection:** Best validation MAE per model; saved to `best_hyperparams.json`.",
        "4. **Final evaluation:** Run `run_forecast` — trains on all data before test week using best params.",
        "",
        "## Phase 1 results",
        "",
    ]
    p1 = df[df["phase"] == 1]
    if len(p1):
        lines.append("```")
        lines.append(
            p1[["experiment_id", "model", "val_mae", "val_rmse", "params"]].to_string(index=False)
        )
        lines.append("```")
    else:
        lines.append("_No phase 1 runs._")

    lines.extend(["", "## Phase 2 — grid search summary", ""])
    p2 = df[df["phase"] == 2]
    if len(p2):
        summary = (
            p2.groupby("model")[["val_mae", "val_rmse"]]
            .agg(["min", "mean"])
            .round(4)
        )
        lines.append("```")
        lines.append(summary.to_string())
        lines.append("```")
        lines.append("")
        lines.append("### Iterative reasoning (for your report)")
        lines.append("")
        for model in p2["model"].unique():
            msub = p2[p2["model"] == model].sort_values("val_mae")
            top = msub.iloc[0]
            worst = msub.iloc[-1]
            lines.append(
                f"- **{model.upper()}:** Best experiment `{top['experiment_id']}` "
                f"(MAE={top['val_mae']:.4f}) vs worst in grid MAE={worst['val_mae']:.4f}. "
                f"Selected params: `{top['params']}`."
            )
    else:
        lines.append("_No phase 2 runs._")

    lines.extend(["", "## Selected hyperparameters (final)", ""])
    lines.append("```json")
    lines.append(json.dumps(best, indent=2))
    lines.append("```")

    path.write_text("\n".join(lines), encoding="utf-8")


def load_best_params() -> dict | None:
    path = Path(settings.BEST_PARAMS_PATH)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)
