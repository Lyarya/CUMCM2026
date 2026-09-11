"""Core mathematics, causality, determinism, and artifact tests for Stage 2A++."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.forecasting.final_models import (
    CausalScaler,
    DLinear,
    HankelPolynomialExpert,
    ResidualMLP,
    StructuralResidualHybrid,
)


TABLE_DIR = Path(__file__).resolve().parents[1] / "results" / "tables" / "forecasting"


def test_dlinear_decomposition_shape_and_identity() -> None:
    torch.manual_seed(2026)
    model = DLinear(21, 7, 5)
    x = torch.randn(4, 21, 1)
    seasonal, trend = model.decompose(x)
    assert seasonal.shape == trend.shape == x.shape
    torch.testing.assert_close(seasonal + trend, x)
    assert model(x).shape == (4, 7, 1)


def test_dlinear_is_deterministic_with_fixed_seed() -> None:
    torch.manual_seed(2026)
    first = DLinear(21, 7, 5)
    torch.manual_seed(2026)
    second = DLinear(21, 7, 5)
    x = torch.linspace(-1, 1, 21).reshape(1, 21, 1)
    torch.testing.assert_close(first(x), second(x))


def test_causal_scaler_round_trip_and_future_independence() -> None:
    past = np.arange(10, dtype=float)
    scaler = CausalScaler.fit(past)
    altered_future = np.r_[past, np.full(10, 999999.0)]
    second = CausalScaler.fit(altered_future[: len(past)])
    assert scaler == second
    np.testing.assert_allclose(scaler.inverse(scaler.transform(past)), past)


def test_hankel_teacher_and_hybrid_are_finite_and_additive() -> None:
    history = 100 + 40 * np.sin(np.arange(432) * 2 * np.pi / 144)
    expert = HankelPolynomialExpert(lookback=432, horizon=144, hankel_rows=144)
    analytical = expert.forecast(history)
    assert analytical.forecast.shape == (144,)
    assert analytical.reconstructed_history.shape == history.shape
    assert np.isfinite(analytical.forecast).all()
    residual = ResidualMLP(432, 144, hidden=4)
    for parameter in residual.parameters():
        torch.nn.init.zeros_(parameter)
    scaler = CausalScaler(0.0, 1.0)
    hybrid = StructuralResidualHybrid(expert, residual, scaler)
    np.testing.assert_allclose(hybrid.predict(history), analytical.forecast, atol=1e-6)


def test_fusion_endpoint_identity() -> None:
    phase = np.arange(8, dtype=float)
    hybrid = phase[::-1]
    np.testing.assert_array_equal(1.0 * phase + 0.0 * hybrid, phase)
    np.testing.assert_array_equal(0.0 * phase + 1.0 * hybrid, hybrid)


def test_final_artifacts_enforce_fixed_causal_protocol() -> None:
    scaling = pd.read_csv(TABLE_DIR / "dlinear_scaling_audit.csv", parse_dates=[
        "scaler_max_timestamp", "training_target_max_timestamp", "target_start_timestamp"
    ])
    assert scaling["causal_assertions_passed"].all()
    assert (scaling["scaler_max_timestamp"] < scaling["target_start_timestamp"]).all()
    assert (scaling["training_target_max_timestamp"] < scaling["target_start_timestamp"]).all()
    assert set(scaling["origin_date"]) == set(pd.date_range("2025-01-18", "2025-01-31").strftime("%Y-%m-%d"))

    selection = pd.read_csv(TABLE_DIR / "forecast_selection_summary.csv")
    assert selection.loc[0, "selection_period"] == "2025-01-18/2025-01-31"
    assert selection.loc[0, "formal_period"] == "2025-02-01/2025-12-31"
    assert bool(selection.loc[0, "formal_model_and_scaler_frozen"])
    assert not bool(selection.loc[0, "online_learning"])
    assert not bool(selection.loc[0, "official_attachment3_used"])

    comparison = pd.read_csv(TABLE_DIR / "forecast_model_final_comparison.csv")
    assert len(comparison) == 9
    assert comparison["selected_by_january"].sum() == 1
    assert set(comparison["model"]) == {
        "Yesterday", "Last Week", "7-day same-slot mean", "Seasonal/Phase",
        "SSA best config", "Polynomial analytical-only teacher", "DLinear",
        "StructuralResidualHybrid", "PhaseStructuralFusion",
    }


def test_formal_predictions_and_protected_hashes_are_valid() -> None:
    predictions = pd.read_csv(TABLE_DIR / "forecast_model_final_predictions.csv")
    assert len(predictions) == 334 * 144
    assert predictions["operating_date"].min() == "2025-02-01"
    assert predictions["operating_date"].max() == "2025-12-31"
    assert predictions.groupby("operating_date").size().eq(144).all()
    numeric = predictions.select_dtypes(include=[np.number])
    assert np.isfinite(numeric).all().all()
    assert (predictions.drop(columns=["operating_date", "datetime", "actual_generation"]) >= 0).all().all()

    audit = json.loads((TABLE_DIR / "forecast_final_leakage_audit.json").read_text(encoding="utf-8"))
    assert audit["protected_artifacts_unchanged"] is True
    assert audit["raw_excel_hashes_unchanged"] is True
    assert audit["protected_artifact_hashes_before"] == audit["protected_artifact_hashes_after"]
    assert audit["raw_excel_hashes_before"] == audit["raw_excel_hashes_after"]
    assert audit["ready_for_stage2b"] is True
