from pathlib import Path

from django.conf import settings

"""
Functions that manage the paths to the data and outputs of the project.
"""

def raw_data_files() -> list[Path]:
    """
    Returns a list of paths to the raw data files.
    """
    root = Path(settings.BASE_DIR)
    files = sorted(root.glob(settings.RAW_DATA_GLOB))
    return [f for f in files if f.is_file()]


def ensure_dirs() -> None:
    """Function to create directories of the paths if not present"""
    settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    settings.SERIES_DIR.mkdir(parents=True, exist_ok=True)
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (settings.OUTPUT_DIR / "eda").mkdir(parents=True, exist_ok=True)
    (settings.OUTPUT_DIR / "forecast").mkdir(parents=True, exist_ok=True)
