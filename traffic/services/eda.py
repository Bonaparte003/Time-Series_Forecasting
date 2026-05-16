"""Exploratory data analysis plots for Task 2."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from django.conf import settings
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from traffic.services.etl import load_metadata, load_square_series
from traffic.services.loader import load_parquet_dataset


def _out(name: str) -> Path:
    path = Path(settings.OUTPUT_DIR) / "eda" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_traffic_pdf(df: pd.DataFrame) -> Path:
    totals = df.groupby("square_id")["internet_traffic"].sum()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(totals, kde=True, stat="density", ax=ax)
    ax.set_xlabel("Total two-month internet traffic (CDR proxy)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of total traffic across 10,000 squares")
    out = _out("01_traffic_pdf.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_three_areas_two_weeks(series_by_square: dict[int, pd.Series]) -> Path:
    start = pd.Timestamp("2013-11-01", tz="UTC")
    end = pd.Timestamp(settings.EDA_TWO_WEEKS_END, tz="UTC") + pd.Timedelta(days=1)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for ax, (sid, s) in zip(axes, series_by_square.items()):
        window = s[(s.index >= start) & (s.index < end)]
        ax.plot(window.index, window.values, linewidth=0.8)
        ax.set_ylabel("Traffic")
        ax.set_title(f"Square {sid}")
    axes[-1].set_xlabel("Time")
    fig.suptitle("Network traffic — first two weeks (Nov 2013)")
    out = _out("02_three_areas_two_weeks.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_stationarity(series: pd.Series, square_id: int) -> Path:
    roll = series.rolling(144)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(series.index, series.values, label="Traffic", linewidth=0.6)
    axes[0].plot(roll.mean().index, roll.mean().values, label="Rolling mean (1 day)")
    axes[0].legend()
    axes[0].set_title(f"Square {square_id} — rolling mean")
    axes[1].plot(roll.std().index, roll.std().values, color="orange", label="Rolling std")
    axes[1].legend()
    axes[1].set_title("Rolling standard deviation")
    out = _out(f"03_stationarity_square_{square_id}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

    adf_stat, pval, *_ = adfuller(series.dropna().values)
    with open(_out(f"03_adf_square_{square_id}.txt"), "w") as f:
        f.write(f"ADF statistic: {adf_stat:.4f}\n")
        f.write(f"p-value: {pval:.6f}\n")
    return out


def plot_decomposition(series: pd.Series, square_id: int) -> Path:
    period = settings.SEASONAL_PERIOD
    res = seasonal_decompose(series, model="additive", period=period)
    fig = res.plot()
    fig.set_size_inches(12, 8)
    fig.suptitle(f"Decomposition — square {square_id} (period={period})")
    out = _out(f"04_decomposition_square_{square_id}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_acf_pacf(series: pd.Series, square_id: int) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    plot_acf(series.dropna(), lags=72, ax=axes[0])
    axes[0].set_title("ACF")
    plot_pacf(series.dropna(), lags=72, ax=axes[1], method="ywm")
    axes[1].set_title("PACF")
    fig.suptitle(f"Autocorrelation — square {square_id}")
    out = _out(f"05_acf_pacf_square_{square_id}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_spatial_heatmap(df: pd.DataFrame) -> Path:
    totals = df.groupby("square_id")["internet_traffic"].sum()
    grid = np.zeros((100, 100))
    for sid, val in totals.items():
        r, c = divmod(int(sid) - 1, 100)
        if 0 <= r < 100 and 0 <= c < 100:
            grid[r, c] = val
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(grid, aspect="auto", cmap="YlOrRd")
    ax.set_title("Spatial heatmap — total two-month traffic")
    ax.set_xlabel("Grid column")
    ax.set_ylabel("Grid row")
    fig.colorbar(im, ax=ax, label="Total traffic")
    out = _out("06_spatial_heatmap.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def run_eda() -> list[Path]:
    df = load_parquet_dataset()
    metadata = load_metadata()
    targets = metadata["target_square_ids"]
    series_map = {sid: load_square_series(sid) for sid in targets}

    outputs = [
        plot_traffic_pdf(df),
        plot_three_areas_two_weeks(series_map),
        plot_spatial_heatmap(df),
    ]
    ref_square = targets[0]
    for sid in targets:
        s = series_map[sid]
        outputs.append(plot_stationarity(s, sid))
        if sid == ref_square:
            outputs.append(plot_decomposition(s, sid))
            outputs.append(plot_acf_pacf(s, sid))

    z = (df["internet_traffic"] - df["internet_traffic"].mean()) / df[
        "internet_traffic"
    ].std()
    outliers = df.loc[z.abs() > 4, ["square_id", "timestamp", "internet_traffic"]]
    outliers.head(500).to_csv(_out("07_outlier_sample.csv"), index=False)
    return outputs
