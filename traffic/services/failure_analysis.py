"""Identify poor forecast intervals and export residual diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from django.conf import settings


def _predictions_dir() -> Path:
    return Path(settings.OUTPUT_DIR) / "forecast" / "predictions"


def save_predictions(
    model_name: str,
    square_id: int,
    index: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Path:
    out = _predictions_dir()
    out.mkdir(parents=True, exist_ok=True)
    residuals = y_true - y_pred
    df = pd.DataFrame(
        {
            "timestamp": index,
            "actual": y_true,
            "predicted": y_pred,
            "residual": residuals,
            "abs_error": np.abs(residuals),
        }
    )
    path = out / f"{model_name}_square_{square_id}.csv"
    df.to_csv(path, index=False)
    return path


def run_failure_analysis(*, window_intervals: int = 36) -> dict:
    """
    window_intervals: length of sliding window in 10-min steps (36 ≈ 6 hours).
    """
    pred_dir = _predictions_dir()
    out_dir = Path(settings.FAILURE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pred_dir.exists():
        raise FileNotFoundError(
            f"No predictions in {pred_dir}. Run: python manage.py run_forecast"
        )

    findings: list[dict] = []

    for csv_path in sorted(pred_dir.glob("*.csv")):
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        stem = csv_path.stem
        parts = stem.split("_square_")
        model_name = parts[0]
        square_id = int(parts[1])

        # Sliding window mean absolute error
        df["window_mae"] = (
            df["abs_error"].rolling(window_intervals, min_periods=1).mean()
        )
        worst_idx = df["window_mae"].idxmax()
        worst_row = df.loc[worst_idx]
        window_start = max(0, worst_idx - window_intervals + 1)
        window = df.iloc[window_start : worst_idx + 1]

        finding = {
            "model": model_name,
            "square_id": square_id,
            "worst_window_start": str(window["timestamp"].iloc[0]),
            "worst_window_end": str(window["timestamp"].iloc[-1]),
            "window_mae": float(worst_row["window_mae"]),
            "peak_abs_error": float(df["abs_error"].max()),
            "overall_mae": float(df["abs_error"].mean()),
        }
        findings.append(finding)

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        axes[0].plot(df["timestamp"], df["actual"], label="Actual", linewidth=1)
        axes[0].plot(df["timestamp"], df["predicted"], label="Predicted", alpha=0.85)
        axes[0].set_title(f"Test week — {model_name.upper()} square {square_id}")
        axes[0].legend()

        axes[1].plot(df["timestamp"], df["residual"], color="crimson", linewidth=0.9)
        axes[1].axhline(0, color="black", linewidth=0.5)
        axes[1].set_ylabel("Residual")
        axes[1].set_title("Residuals (actual − predicted)")

        fig.tight_layout()
        fig.savefig(out_dir / f"residuals_{model_name}_square_{square_id}.png", dpi=150)
        plt.close(fig)

        # Zoom worst window
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(window["timestamp"], window["actual"], label="Actual", marker="o", ms=3)
        ax.plot(window["timestamp"], window["predicted"], label="Predicted", marker="o", ms=3)
        ax.set_title(
            f"Worst ~{window_intervals * 10 // 60}h window "
            f"(MAE={finding['window_mae']:.3f}) — {model_name} square {square_id}"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            out_dir / f"worst_window_{model_name}_square_{square_id}.png", dpi=150
        )
        plt.close(fig)

    report = {
        "window_intervals": window_intervals,
        "window_description": f"{window_intervals} × 10-min intervals",
        "findings": findings,
        "overall_worst": max(findings, key=lambda x: x["window_mae"]) if findings else None,
    }

    json_path = out_dir / "failure_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_path = out_dir / "failure_analysis.md"
    _write_failure_markdown(report, md_path)

    return {"output_dir": str(out_dir), "report": str(json_path), "markdown": str(md_path)}


def _write_failure_markdown(report: dict, path: Path) -> None:
    lines = [
        "# Failure Analysis (auto-generated)",
        "",
        "Use this draft for **Task 3 Failure Analysis** in your report; add interpretation.",
        "",
        f"Sliding window: {report['window_description']}.",
        "",
        "## Per model / square",
        "",
    ]
    for f in report["findings"]:
        lines.append(
            f"### {f['model'].upper()} — square {f['square_id']}\n"
            f"- Worst window: **{f['worst_window_start']}** → **{f['worst_window_end']}**\n"
            f"- Window MAE: **{f['window_mae']:.4f}**\n"
            f"- Peak absolute error: **{f['peak_abs_error']:.4f}**\n"
            f"- See: `worst_window_{f['model']}_square_{f['square_id']}.png`\n"
        )

    ow = report.get("overall_worst")
    if ow:
        lines.extend(
            [
                "## Suggested focus for report",
                "",
                f"Largest sustained error: **{ow['model'].upper()}** on square "
                f"**{ow['square_id']}** during **{ow['worst_window_start']}** – "
                f"**{ow['worst_window_end']}**.",
                "",
                "Possible causes to discuss (link to Task 2):",
                "- Pre-holiday traffic surge (mid-December).",
                "- Model lag on sharp ramps (neural nets using recursive preds).",
                "- Non-stationarity / weekend vs weekday pattern shift.",
                "- Single-square idiosyncrasy vs city-wide events.",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
