"""Mathematical, numerical and leakage tests for the Stage 2A+ SSA checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.forecasting.ssa import SSARecurrentForecaster, diagonal_average, trajectory_matrix
from src.forecasting.ssa_checkpoint import (
    causal_history,
    paired_daily_comparison,
    select_best_ssa,
)


RESULT_DIR = Path(__file__).resolve().parents[1] / "results" / "tables" / "forecasting"


def test_ssa_trajectory_matrix_dimensions_and_values() -> None:
    values = np.arange(8, dtype=float)
    matrix = trajectory_matrix(values, 3)
    assert matrix.shape == (3, 6)
    np.testing.assert_array_equal(matrix[:, 0], [0, 1, 2])
    np.testing.assert_array_equal(matrix[:, -1], [5, 6, 7])


def test_ssa_diagonal_averaging_is_exact() -> None:
    matrix = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    np.testing.assert_allclose(diagonal_average(matrix), [1.0, 2.0, 3.0, 4.0])


def test_ssa_rank_is_configurable_and_must_be_legal() -> None:
    history = 10 + np.sin(np.arange(432) * 2 * np.pi / 144)
    forecaster = SSARecurrentForecaster(horizon=144, embedding_dimension=144)
    results = forecaster.forecast_ranks(history, (1, 3, 5))
    assert set(results) == {1, 3, 5}
    with pytest.raises(ValueError):
        forecaster.forecast_ranks(history, (145,))


def test_ssa_recurrent_coefficients_and_forecast_are_finite() -> None:
    history = 3000 + 1800 * np.sin(np.arange(1008) * 2 * np.pi / 144)
    result = SSARecurrentForecaster().forecast_ranks(history, (2,))[2]
    assert result.status == "ok"
    assert result.forecast.shape == (144,)
    assert np.isfinite(result.forecast).all()
    assert result.recurrence_denominator > 0
    assert np.isfinite(result.recurrence_coefficient_norm)


def test_ssa_rejects_nan_or_inf_instead_of_silent_propagation() -> None:
    history = np.ones(432)
    history[10] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        SSARecurrentForecaster().forecast_ranks(history, (1,))


def test_causal_window_uses_only_days_strictly_before_target() -> None:
    values = np.arange(20 * 144, dtype=float).reshape(20, 144)
    before = causal_history(values, 14, window_days=7)
    altered = values.copy()
    altered[14:] = -999_999
    after = causal_history(altered, 14, window_days=7)
    np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(before, values[7:14].reshape(-1))


def test_ssa_selection_refuses_non_january_rows() -> None:
    valid = pd.DataFrame(
        {
            "scope": ["january_common_origin_validation"],
            "selection_eligible": [True],
            "full_rmse_kw": [10.0],
            "window_days": [7],
            "rank": [3],
        }
    )
    assert select_best_ssa(valid) == (7, 3)
    invalid = valid.assign(scope="february_december_walk_forward")
    with pytest.raises(ValueError, match="January"):
        select_best_ssa(invalid)


def test_daily_paired_comparison_requires_same_days_and_shapes() -> None:
    dates = pd.date_range("2025-02-01", periods=2, freq="D")
    actual = np.ones((2, 144))
    table = paired_daily_comparison(dates, actual, actual * 0.9, actual * 1.1)
    assert list(table["date"]) == ["2025-02-01", "2025-02-02"]
    assert len(table) == 2
    with pytest.raises(ValueError, match="identical shapes"):
        paired_daily_comparison(dates, actual, np.ones((1, 144)), actual)


def test_formal_ssa_artifacts_freeze_january_choice_and_exclude_forbidden_inputs() -> None:
    selection = pd.read_csv(RESULT_DIR / "ssa_model_selection.csv")
    assert len(selection) == 1
    assert selection.loc[0, "selection_period"] == "2025-01-15/2025-01-31"
    assert selection.loc[0, "formal_evaluation_period"] == "2025-02-01/2025-12-31"
    assert bool(selection.loc[0, "formal_parameters_fixed"])
    assert not bool(selection.loc[0, "online_learning"])
    assert not bool(selection.loc[0, "official_forecast_used"])

    audit = json.loads((RESULT_DIR / "ssa_leakage_audit.json").read_text(encoding="utf-8"))
    assert audit["selection_uses_january_only"] is True
    assert audit["formal_parameters_fixed"] is True
    assert audit["history_end_strictly_before_target_date"] is True
    assert audit["mean_centering"] is True
    assert audit["standardization"] is False
    assert audit["full_year_mean_or_std_used"] is False
    assert audit["official_forecast_used"] is False
    assert audit["online_learning"] is False
    assert audit["paired_comparison_unit"] == "forecast_day"
    assert audit["paired_day_count"] == 334

    predictions = pd.read_csv(RESULT_DIR / "q2_ssa_predictions.csv")
    assert len(predictions) == 334 * 144
    assert predictions["operating_date"].min() == "2025-02-01"
    assert predictions["operating_date"].max() == "2025-12-31"
    assert predictions.groupby("operating_date").size().eq(144).all()
    assert np.isfinite(predictions["ssa_pred"]).all()
