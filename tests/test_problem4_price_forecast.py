"""Chronological selection and future-leakage tests for price forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.problem4.price_data import build_price_canonical
from src.problem4.price_forecast import (
    EVALUATION_START,
    SELECTED_OUTPUT_COLUMN,
    build_price_forecasts,
)


def test_lightweight_price_forecast_is_selected_on_january_only() -> None:
    predictions, comparison, selection = build_price_forecasts(build_price_canonical())
    assert set(comparison["method"]) == {"前一日同刻", "前一周同刻", "近7日同刻均值", "因果岭回归"}
    assert selection["validation_period"] == ["2025-01-17", "2025-01-30"]
    assert selection["evaluation_period"] == ["2025-02-01", "2025-12-31"]
    assert selection["selected_method"] == "因果岭回归"
    assert selection["selected_ridge_alpha"] == 0.01
    assert selection["all_price_lags_strictly_historical"] is True
    assert selection["model_frozen_before_evaluation"] is True
    formal = predictions.loc[predictions["operating_date"].ge(EVALUATION_START)]
    assert len(formal) == 334 * 144
    assert np.isfinite(formal[SELECTED_OUTPUT_COLUMN]).all()


def test_changing_future_actual_prices_cannot_change_already_issued_forecast() -> None:
    canonical = build_price_canonical()
    first, _, _ = build_price_forecasts(canonical)
    issue = pd.Timestamp("2025-06-20 00:00")
    target_day = pd.Timestamp("2025-06-20")
    altered = canonical.copy()
    altered.loc[altered["interval_end"].gt(issue), "price_yuan_per_kwh"] = 999.0
    second, _, _ = build_price_forecasts(altered)
    day_mask = first["operating_date"].eq(target_day)
    np.testing.assert_allclose(
        first.loc[day_mask, SELECTED_OUTPUT_COLUMN],
        second.loc[day_mask, SELECTED_OUTPUT_COLUMN],
        rtol=0,
        atol=1e-12,
    )


def test_missing_timestamp_rejects_positional_lag_construction() -> None:
    canonical = build_price_canonical().drop(index=144)
    with pytest.raises(ValueError, match="complete, unique"):
        build_price_forecasts(canonical)


def test_reported_forecast_metrics_match_generated_results() -> None:
    _, comparison, selection = build_price_forecasts(build_price_canonical())
    selected = comparison.loc[comparison["method"].eq(selection["selected_method"])].set_index("period")
    assert selected.loc["validation", "rmse_yuan_per_kwh"] == pytest.approx(0.0485596924, abs=1e-10)
    assert selected.loc["evaluation", "rmse_yuan_per_kwh"] == pytest.approx(0.0624482958, abs=1e-10)
