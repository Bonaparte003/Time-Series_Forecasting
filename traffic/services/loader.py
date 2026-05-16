"""Chunked loading and memory-optimized parsing of raw TIM text files."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
from django.conf import settings


COLUMNS = ["square_id", "time_ms", "country_code", "internet_traffic"]
USECOLS = [0, 1, 2, settings.INTERNET_COL]


def memory_usage_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum()) / (1024**2)


def sample_raw_frame(path: Path, nrows: int = 100_000) -> pd.DataFrame:
    """Load a small sample with default dtypes (for before/after comparison)."""
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        usecols=USECOLS,
        nrows=nrows,
        names=COLUMNS,
    )
    return df


def optimize_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["square_id"] = out["square_id"].astype(np.uint16)
    out["time_ms"] = out["time_ms"].astype(np.int64)
    out["country_code"] = out["country_code"].astype(np.int16)
    out["internet_traffic"] = out["internet_traffic"].astype(np.float32)
    return out


def iter_file_chunks(path: Path, chunksize: int | None = None):
    chunksize = chunksize or settings.CHUNK_SIZE
    reader = pd.read_csv(
        path,
        sep="\t",
        header=None,
        usecols=USECOLS,
        names=COLUMNS,
        chunksize=chunksize,
        dtype={
            "square_id": np.uint16,
            "time_ms": np.int64,
            "country_code": np.int16,
            "internet_traffic": np.float32,
        },
    )
    for chunk in reader:
        filtered = chunk[chunk["country_code"] == settings.COUNTRY_ITALY].drop(
            columns=["country_code"]
        )
        if not filtered.empty:
            yield filtered


def ingest_files(
    files: list[Path],
    daily_dir: Path,
    max_files: int | None = None,
) -> dict:
    """
    Stream raw files into one Parquet file per day under daily_dir.
    Returns stats including memory before/after dtype optimization on a sample.
    """
    if max_files:
        files = files[:max_files]

    if not files:
        raise FileNotFoundError(
            f"No raw files matched {settings.RAW_DATA_GLOB} under {settings.BASE_DIR}"
        )

    sample_path = files[0]
    raw_sample = sample_raw_frame(sample_path)
    mem_before = memory_usage_mb(raw_sample)
    mem_after = memory_usage_mb(optimize_frame(raw_sample))

    daily_dir.mkdir(parents=True, exist_ok=True)
    total_rows = 0

    for path in files:
        parts: list[pd.DataFrame] = []
        for chunk in iter_file_chunks(path):
            parts.append(chunk)
        if not parts:
            continue
        day_df = pd.concat(parts, ignore_index=True)
        day_df["timestamp"] = pd.to_datetime(day_df["time_ms"], unit="ms", utc=True)
        day_df = day_df.drop(columns=["time_ms"])
        day_df = day_df.groupby(["square_id", "timestamp"], as_index=False)[
            "internet_traffic"
        ].sum()
        out_file = daily_dir / f"{path.stem}.parquet"
        day_df.to_parquet(out_file, index=False)
        total_rows += len(day_df)
        del parts, day_df
        gc.collect()

    return {
        "files_processed": len(files),
        "rows_written": total_rows,
        "memory_before_mb": mem_before,
        "memory_after_mb": mem_after,
        "sample_path": str(sample_path),
        "daily_dir": str(daily_dir),
    }


def load_parquet_dataset(daily_dir: Path | None = None) -> pd.DataFrame:
    """Load all daily Parquet shards into one dataframe."""
    daily_dir = daily_dir or (settings.PROCESSED_DIR / "daily")
    files = sorted(daily_dir.glob("*.parquet"))
    if not files:
        merged = settings.PARQUET_PATH
        if merged.exists():
            return pd.read_parquet(merged)
        raise FileNotFoundError(
            f"No parquet data in {daily_dir}. Run: python manage.py ingest_raw"
        )
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df
