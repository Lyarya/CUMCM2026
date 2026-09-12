"""Control-variable and identity checks for the independent Q4 robustness layer."""

from __future__ import annotations

import numpy as np
import pytest

from src.problem2.forecast_interface import get_q2_day_inputs
from src.problem4.run_robustness import (
    PRICE_GAMMAS,
    RobustnessSpec,
    UNCERTAINTY_KAPPAS,
    formal_baseline,
    formal_baseline_payload,
    load_formal_price_frame,
    run_case,
    scale_price_frame,
    scale_q2_scenarios,
)


def test_price_volatility_transform_changes_only_centered_amplitude() -> None:
    formal = load_formal_price_frame()
    original = formal["price_yuan_per_kwh"].to_numpy(float)
    mean = float(original.mean())
    for gamma in PRICE_GAMMAS:
        transformed = scale_price_frame(formal, gamma)
        price = transformed["price_yuan_per_kwh"].to_numpy(float)
        np.testing.assert_allclose(price - mean, gamma * (original - mean), atol=1e-12)
        assert price.mean() == pytest.approx(mean, abs=1e-12)
        assert np.isfinite(price).all()
        assert (price >= 0.0).all()
        np.testing.assert_array_equal(
            transformed[["operating_date", "slot", "interval_end"]],
            formal[["operating_date", "slot", "interval_end"]],
        )
    assert scale_price_frame(formal, 0.0)["price_yuan_per_kwh"].nunique() == 1
    np.testing.assert_array_equal(
        scale_price_frame(formal, 1.0)["price_yuan_per_kwh"],
        formal["price_yuan_per_kwh"],
    )


def test_uncertainty_transform_preserves_centers_pairing_and_causality() -> None:
    base = get_q2_day_inputs("2025-06-20", initial_energy=6000.0)
    for kappa in UNCERTAINTY_KAPPAS:
        transformed = scale_q2_scenarios(base, kappa)
        np.testing.assert_array_equal(transformed.load_forecast, base.load_forecast)
        np.testing.assert_array_equal(transformed.pv_forecast, base.pv_forecast)
        np.testing.assert_array_equal(
            transformed.scenario_source_dates, base.scenario_source_dates
        )
        assert transformed.load_scenarios.shape == transformed.pv_scenarios.shape == (50, 144)
        assert np.isfinite(transformed.load_scenarios).all()
        assert np.isfinite(transformed.pv_scenarios).all()
        assert (transformed.pv_scenarios >= 0.0).all()
        assert (
            transformed.scenario_source_dates < np.datetime64(base.date)
        ).all()
    identity = scale_q2_scenarios(base, 1.0)
    np.testing.assert_array_equal(identity.load_scenarios, base.load_scenarios)
    np.testing.assert_array_equal(identity.pv_scenarios, base.pv_scenarios)


def test_formal_baseline_is_read_from_locked_q4_results() -> None:
    baseline = formal_baseline()
    assert baseline["q42_realized_total_cost_yuan"] == pytest.approx(
        15547956.356361, abs=1e-6
    )
    assert baseline["q42_emergency_energy_kwh"] == pytest.approx(
        396204.272796, abs=1e-6
    )
    assert baseline["s2_realized_total_cost_yuan"] == pytest.approx(
        15196922.472637, abs=1e-6
    )
    assert baseline["s2_emergency_energy_kwh"] == pytest.approx(
        302154.909779, abs=1e-6
    )
    assert baseline["s2_adjustment_cost_yuan"] == pytest.approx(8799.979138, abs=1e-6)
    payload = formal_baseline_payload()
    assert payload["baseline_reproduction"] == "REUSED_AUDITED_FORMAL_BASELINE"
    assert payload["audit_status"] == "PASS"


def test_annual_identity_case_is_prohibited() -> None:
    with pytest.raises(ValueError, match="annual identity rerun is prohibited"):
        run_case(RobustnessSpec(1.0, 1.0))
