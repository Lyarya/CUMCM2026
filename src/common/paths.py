"""Canonical project paths shared by preprocessing, models and figures."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RESULTS_DIR = PROJECT_ROOT / "results"
PAPER_DIR = PROJECT_ROOT / "paper"
PAPER_FIGURES_DIR = PAPER_DIR / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"


def problem_results_dir(problem: int | str) -> Path:
    """Return the standard results directory for one problem."""
    label = str(problem).lower().removeprefix("problem")
    if not label.isdigit():
        raise ValueError(f"Problem identifier must be numeric, got {problem!r}.")
    return RESULTS_DIR / f"problem{label}"


def problem_figures_dir(problem: int | str) -> Path:
    """Return ``results/problemX/figures`` without creating it."""
    return problem_results_dir(problem) / "figures"


def paper_problem_figures_dir(problem: int | str) -> Path:
    """Return ``paper/figures/problemX`` without creating it."""
    label = str(problem).lower().removeprefix("problem")
    if not label.isdigit():
        raise ValueError(f"Problem identifier must be numeric, got {problem!r}.")
    return PAPER_FIGURES_DIR / f"problem{label}"
