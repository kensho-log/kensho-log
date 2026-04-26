"""simulate_dca_loss_horizon / simulate_dca_loss_horizon_batch の単体テスト。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kensho_log.simulations.dca_loss_horizon import (
    InsufficientDataError,
    LossHorizonResult,
    simulate_dca_loss_horizon,
    simulate_dca_loss_horizon_batch,
)


def _series(prices: list[float], start: str = "2000-01-31") -> pd.Series:
    idx = pd.date_range(start=start, periods=len(prices), freq="ME")
    return pd.Series(prices, index=idx, name="close")


class TestValidation:
    def test_non_series_raises(self):
        with pytest.raises(TypeError):
            simulate_dca_loss_horizon([1.0, 2.0], "2000-01-31")  # type: ignore[arg-type]

    def test_empty_series_raises(self):
        s = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
        with pytest.raises(ValueError):
            simulate_dca_loss_horizon(s, "2000-01-31")

    def test_non_datetime_index_raises(self):
        s = pd.Series([100.0, 101.0], index=[0, 1])
        with pytest.raises(TypeError):
            simulate_dca_loss_horizon(s, "2000-01-31")

    def test_negative_price_raises(self):
        s = _series([100.0, -1.0])
        with pytest.raises(ValueError):
            simulate_dca_loss_horizon(s, "2000-01-31")

    def test_zero_amount_raises(self):
        s = _series([100.0] * 12)
        with pytest.raises(ValueError):
            simulate_dca_loss_horizon(s, "2000-01-31", monthly_amount=0.0)

    def test_start_after_last_raises(self):
        s = _series([100.0] * 12, start="2000-01-31")
        with pytest.raises(InsufficientDataError):
            simulate_dca_loss_horizon(s, "2099-12-31")


class TestFlatMarket:
    def test_flat_never_profits_strictly(self):
        s = _series([100.0] * 24)
        r = simulate_dca_loss_horizon(s, "2000-01-31", monthly_amount=1.0)
        assert isinstance(r, LossHorizonResult)
        assert r.reached_profit is False
        assert r.months_to_profit is None
        assert r.years_to_profit is None
        assert r.final_invested == pytest.approx(24.0)
        assert r.final_value == pytest.approx(24.0)
        assert r.final_ratio == pytest.approx(1.0)


class TestMonotonicUp:
    def test_up_reaches_profit_at_month_1(self):
        s = _series([100.0, 110.0, 120.0, 130.0])
        r = simulate_dca_loss_horizon(s, "2000-01-31", monthly_amount=1.0)
        assert r.reached_profit is True
        assert r.months_to_profit == 1
        assert r.years_to_profit == pytest.approx(1 / 12)
        assert r.profit_month == pd.Timestamp("2000-02-29")

    def test_up_after_initial_dip(self):
        prices = [100.0, 90.0, 90.0, 110.0, 120.0]
        s = _series(prices)
        r = simulate_dca_loss_horizon(s, "2000-01-31", monthly_amount=1.0)
        assert r.reached_profit is True
        assert r.months_to_profit in (3, 4)
        assert r.profit_month == s.index[r.months_to_profit]


class TestMonotonicDown:
    def test_down_never_reaches_profit(self):
        s = _series([100.0, 90.0, 80.0, 70.0, 60.0])
        r = simulate_dca_loss_horizon(s, "2000-01-31", monthly_amount=1.0)
        assert r.reached_profit is False
        assert r.final_ratio < 1.0


class TestStartMonthResolution:
    def test_start_aligns_to_first_month_on_or_after(self):
        s = _series([100.0, 110.0, 120.0], start="2000-01-31")
        r = simulate_dca_loss_horizon(s, "2000-01-15", monthly_amount=1.0)
        assert r.start_month == pd.Timestamp("2000-01-31")
        assert r.months_available == 3

    def test_start_mid_series_skips_prior_months(self):
        s = _series([100.0, 110.0, 120.0, 130.0], start="2000-01-31")
        r = simulate_dca_loss_horizon(s, "2000-03-01", monthly_amount=1.0)
        assert r.start_month == pd.Timestamp("2000-03-31")
        assert r.months_available == 2


class TestAnalyticCase:
    def test_two_month_known_values(self):
        s = _series([10.0, 20.0])
        r = simulate_dca_loss_horizon(s, "2000-01-31", monthly_amount=1.0)
        assert r.final_invested == pytest.approx(2.0)
        assert r.final_value == pytest.approx((1 / 10 + 1 / 20) * 20)
        assert r.reached_profit is True
        assert r.months_to_profit == 1

    def test_first_month_value_equals_invested(self):
        s = _series([10.0] * 5)
        r = simulate_dca_loss_horizon(s, "2000-01-31", monthly_amount=1.0)
        assert r.months_available == 5
        assert r.reached_profit is False


class TestBatch:
    def test_batch_returns_row_per_month(self):
        prices = [100.0, 110.0, 120.0, 130.0]
        s = _series(prices)
        df = simulate_dca_loss_horizon_batch(s, monthly_amount=1.0)
        assert len(df) == len(s)
        assert df.index.name == "start_month"
        assert set(
            [
                "months_available",
                "reached_profit",
                "months_to_profit",
                "final_value",
                "final_ratio",
            ]
        ).issubset(df.columns)

    def test_batch_min_months_filters_tail(self):
        s = _series([100.0] * 36)
        df = simulate_dca_loss_horizon_batch(
            s, monthly_amount=1.0, min_months_available=12
        )
        assert len(df) == 36 - 12 + 1
        assert df.index.max() == s.index[-12]

    def test_batch_all_up_all_reach_profit(self):
        prices = [100.0 + i for i in range(24)]
        s = _series(prices)
        df = simulate_dca_loss_horizon_batch(s, monthly_amount=1.0)
        assert (df["reached_profit"] | (df["months_available"] == 1)).all()

    def test_batch_min_months_validation(self):
        s = _series([100.0] * 5)
        with pytest.raises(ValueError):
            simulate_dca_loss_horizon_batch(s, min_months_available=0)

    def test_batch_empty_when_all_filtered(self):
        s = _series([100.0] * 3)
        df = simulate_dca_loss_horizon_batch(s, min_months_available=10)
        assert df.empty
