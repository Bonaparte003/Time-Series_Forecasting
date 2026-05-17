"""Build per-square time series and assignment metadata."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from django.conf import settings

from traffic.services.loader import load_parquet_dataset
from traffic.services.verbose import LogFn


def series_from_frame(sub: pd.DataFrame) -> pd.Series:
    s = (
        sub.set_index("timestamp")["internet_traffic"]
        .sort_index()
        .astype("float32")
    )
    s = s[~s.index.duplicated(keep="last")]
    full_idx = pd.date_range(s.index.min(), s.index.max(), freq="10min", tz="UTC")
    return s.reindex(full_idx, fill_value=0.0)


def save_series(
    series_map: dict[int, pd.Series],
    out_dir: Path,
    log: LogFn = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for square_id, s in series_map.items():
        path = out_dir / f"square_{square_id}.parquet"
        pd.DataFrame({"traffic": s}).to_parquet(path)
        if log:
            log(f"  wrote {path.name} ({len(s):,} intervals)")


def load_square_series(square_id: int) -> pd.Series:
    path = settings.SERIES_DIR / f"square_{square_id}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Series for square {square_id} not found at {path}")
    df = pd.read_parquet(path)
    return df["traffic"]


def run_build_series(
    parquet_daily_dir: Path | None = None,
    save_all: bool = False,
    log: LogFn = None,
) -> dict:
    df = load_parquet_dataset(parquet_daily_dir, log=log)
    if log:
        log("Computing per-square traffic totals...")
    totals = df.groupby("square_id")["internet_traffic"].sum()
    top_square = int(totals.idxmax())
    target_squares = [top_square] + list(settings.FIXED_SQUARE_IDS)

    if save_all:
        ids_to_build = totals.index.astype(int).tolist()
    else:
        ids_to_build = target_squares

    if log:
        log(f"Top-traffic square: {top_square}")
        log(f"Building {len(ids_to_build)} time series (save_all={save_all})...")

    series_map = {}
    for square_id in ids_to_build:
        sub = df[df["square_id"] == square_id]
        if sub.empty:
            if log:
                log(f"  square {square_id}: skipped (no rows)")
            continue
        if log:
            log(f"  square {square_id}: {len(sub):,} raw rows → regular 10-min series")
        series_map[int(square_id)] = series_from_frame(sub)

    metadata = {
        "top_traffic_square_id": top_square,
        "target_square_ids": target_squares,
        "square_totals_sample": {
            int(k): float(v)
            for k, v in totals.nlargest(10).items()
        },
        "n_squares": int(totals.shape[0]),
    }

    if log:
        log(f"Saving series to {settings.SERIES_DIR}")
    save_series(series_map, settings.SERIES_DIR, log=log)

    settings.METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(settings.METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    if log:
        log(f"Wrote metadata: {settings.METADATA_PATH}")

    return metadata


def load_metadata() -> dict:
    with open(settings.METADATA_PATH, encoding="utf-8") as f:
        return json.load(f)
