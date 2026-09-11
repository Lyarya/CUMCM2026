"""Convenience wrapper for the formal Stage 2A forecasting pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.forecasting.run import main


if __name__ == "__main__":
    main()
