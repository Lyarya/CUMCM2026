"""Focused tests for the frozen Q2 forecast significance audit."""

from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest

import src.problem2.forecast_significance as significance_module
from src.problem2.forecast_significance import (
    ALPHA,
    COMPARATORS,
    OOS_END,
    OOS_START,
    REFERENCE_MODEL,
    SELECTION_PATH,
    daily_mae_table,
    load_frozen_predictions,
    paired_significance,
)


@pytest.fixture(scope="module")
def frozen_frame() -> pd.DataFrame:
    return load_frozen_predictions()


def test_common_oos_period_and_daily_horizon(frozen_frame: pd.DataFrame) -> None:
    assert frozen_frame["operating_date"].min() == OOS_START
    assert frozen_frame["operating_date"].max() == OOS_END
    assert frozen_frame["operating_date"].nunique() == 334
    assert (frozen_frame.groupby("operating_date").size() == 144).all()
    assert not frozen_frame.duplicated(["operating_date", "datetime"]).any()


def test_saved_comparators_are_finite_and_formal_selection_is_unchanged(
    frozen_frame: pd.DataFrame,
) -> None:
    assert np.isfinite(frozen_frame[[REFERENCE_MODEL, *COMPARATORS]].to_numpy(float)).all()
    audit = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    assert audit["selected_generation_forecaster"] == REFERENCE_MODEL
    assert audit["strict_causal_protocol"] is True
    assert audit["formal_model_and_scaler_frozen"] is True
    assert audit["protected_artifacts_unchanged"] is True
    assert audit["raw_excel_hashes_unchanged"] is True


def test_daily_mae_uses_operating_day_as_statistical_unit(frozen_frame: pd.DataFrame) -> None:
    daily = daily_mae_table(frozen_frame)
    assert daily.shape == (334 * (1 + len(COMPARATORS)), 3)
    assert (daily.groupby("model").size() == 334).all()
    first = frozen_frame.loc[frozen_frame["operating_date"] == OOS_START]
    expected = np.mean(np.abs(first[REFERENCE_MODEL] - first["actual_generation"]))
    observed = daily.loc[
        (daily["date"] == OOS_START.date().isoformat()) & (daily["model"] == REFERENCE_MODEL),
        "daily_mae",
    ].item()
    assert observed == pytest.approx(expected)


def test_paired_tests_and_fdr_are_complete(frozen_frame: pd.DataFrame) -> None:
    results = paired_significance(daily_mae_table(frozen_frame))
    assert results["comparator"].tolist() == list(COMPARATORS)
    assert (results["n_days"] == 334).all()
    assert results["raw_p"].between(0.0, 1.0).all()
    assert results["fdr_adjusted_p"].between(0.0, 1.0).all()
    assert results["reference_daily_win_rate"].between(0.0, 1.0).all()
    assert results["tie_rate"].between(0.0, 1.0).all()
    assert ALPHA == pytest.approx(0.05)


def test_evaluation_does_not_construct_or_retrain_forecasts() -> None:
    source = inspect.getsource(significance_module)
    forbidden = ("fit_torch_regressor(", "solve_expected_cost_dispatch(", "run_chronological(")
    assert all(token not in source for token in forbidden)
