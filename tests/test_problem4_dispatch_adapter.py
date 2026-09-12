from __future__ import annotations

import numpy as np
import pandas as pd

from src.problem3.rolling_dispatch import build_release_inputs
from src.problem4.dispatch_adapter import CAUSAL_PRICE, GIVEN_PRICE, q43_price_provider


def test_q3_default_price_path_is_preserved() -> None:
    original, _ = build_release_inputs("2025-02-01", 0, 6000.0)
    explicit, _ = build_release_inputs(
        "2025-02-01", 0, 6000.0, price_yuan_per_kwh=original.price
    )
    assert np.array_equal(original.price, explicit.price)


def test_q4_price_provider_returns_aligned_decision_and_settlement_prices() -> None:
    for mode in (GIVEN_PRICE, CAUSAL_PRICE):
        decision, settlement = q43_price_provider(mode)(pd.Timestamp("2025-02-01"))
        assert decision.shape == settlement.shape == (144,)
        assert np.isfinite(decision).all()
        assert np.isfinite(settlement).all()
        assert (decision >= 0).all()
        assert (settlement >= 0).all()


def test_given_price_decision_equals_settlement() -> None:
    decision, settlement = q43_price_provider(GIVEN_PRICE)(pd.Timestamp("2025-02-01"))
    assert np.array_equal(decision, settlement)
