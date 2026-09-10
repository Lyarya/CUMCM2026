"""Small, explicit data-loading helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_table(path: str | Path, **kwargs: object) -> pd.DataFrame:
    """Load CSV, Excel or Parquet data based on its suffix."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source, **kwargs)
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(source, **kwargs)
    if suffix == ".parquet":
        return pd.read_parquet(source, **kwargs)
    raise ValueError(f"Unsupported table format: {source.suffix}")


def save_table(frame: pd.DataFrame, path: str | Path, **kwargs: object) -> Path:
    """Save a DataFrame using a deterministic, index-free default."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix.lower()
    if suffix == ".csv":
        frame.to_csv(target, index=False, **kwargs)
    elif suffix in {".xls", ".xlsx"}:
        frame.to_excel(target, index=False, **kwargs)
    elif suffix == ".parquet":
        frame.to_parquet(target, index=False, **kwargs)
    else:
        raise ValueError(f"Unsupported table format: {target.suffix}")
    return target
