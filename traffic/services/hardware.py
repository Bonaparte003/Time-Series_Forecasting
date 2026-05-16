"""Collect hardware/software metadata for reproducibility reports."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import django


def collect_environment() -> dict:
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python_version": sys.version,
        "django_version": django.get_version(),
        "machine": platform.machine(),
    }


def save_environment(path: Path) -> dict:
    info = collect_environment()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return info
