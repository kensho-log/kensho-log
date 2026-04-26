"""simulate_age_vs_amount / simulate_age_vs_amount_batch の単体テスト。

- 純粋関数なのでネットワーク不要
- 解析的ケースで厳密検証
- データ不足時の挙動（例外 / skip）を明示的に検証
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kensho_log.simulations.age_vs_amount import (
    AgeVsAmountResult,
    InsufficientDataError,
    simulate_age_vs_amount,
    simulate_age_vs_amount_batch,
)


def _monthly_flat(start: str, periods: int, price: float = 100.0) -> pd.Series:
    idx = pd.date_range(start=start, periods=periods, freq="ME")
    return pd.Series([price] * periods, index=idx, name="close")


def _monthly_monotonic(
    start: str, periods: int, base: float = 100.0, step: float = 1.0
) -> pd.Series:
    idx = pd.date_range(start=start, periods=periods, freq="ME")
    vals = [base + step * i for i in range(periods)]
    return pd.Series(vals, index=idx, name="close")


class TestValidation:
    def test_non_series_raises(self):
        with pytest.raises(TypeError):
            simulate_age_vs_amount([1.0, 2.0], birth_year=1960)  # type: ignore[arg-type]

    def test_non_datetime_index_raises(self):
        s = pd.Series([1.0, 2.0], index=[0, 1])
        with pytest.raises(TypeError):
            simulate_age_vs_amount(s, birth_year=1960)

    def test_negative_price_raises(self):
        idx = pd.date_range("1980-01-31", periods=12, freq="ME")
        s = pd.Series([100.0] * 11 + [-1.0], index=idx)
        with pytest.raises(ValueError):
            simulate_age_vs_amount(s, birth_year=1960)

    def test_age_conflict_raises(self):
        s = _monthly_flat("1980-01-31", 600)
        with pytest.raises(ValueError):
            simulate_age_vs_amount(s, birth_year=1960, a_start_age=70)

    def test_duration_overflows_end_age_raises(self):
        s = _monthly_flat("1980-01-31", 600)
        with pytest.raises(ValueError):
            simulate_age_vs_amount(
                s, birth_year=1960, a_start_age=20, a_duration_years=50, end_age=60
            )

    def test_insufficient_data_start_raises(self):
        s = _monthly_flat("1985-01-31", 12)
        with pytest.raises(InsufficientDataError):
            simulate_age_vs_amount(s, birth_year=1960)

    def test_insufficient_data_end_raises(self):
        s = _monthly_flat("1980-01-31", 12)
        with pytest.raises(InsufficientDataError):
            simulate_age_vs_amount(s, birth_year=1960)


class TestFlatMarket:
    def test_flat_market_equals_invested(self):
        s = _monthly_flat("1980-01-31", 40 * 12 + 12, price=100.0)
        r = simulate_age_vs_amount(s, birth_year=1960)
        assert isinstance(r, AgeVsAmountResult)

        assert r.total_invested_a == pytest.approx(30_000 * 12 * 40)
        assert r.total_invested_b == pytest.approx(100_000 * 12 * 20)
        assert r.final_value_a == pytest.approx(r.total_invested_a, rel=1e-10)
        assert r.final_value_b == pytest.approx(r.total_invested_b, rel=1e-10)

        assert r.winner == "B"
        assert r.diff_pct_a_vs_b < 0
        assert r.diff_pct_a_vs_b == pytest.approx(
            (14_400_000 - 24_000_000) / 24_000_000 * 100
        )


class TestMonotonicUp:
    def test_monotonic_up_A_can_beat_B_via_compounding(self):
        s = _monthly_monotonic("1980-01-31", 40 * 12 + 12, base=100.0, step=10.0)
        r = simulate_age_vs_amount(s, birth_year=1960)
        assert r.winner == "A"
        assert r.diff_pct_a_vs_b > 0
        assert r.total_invested_a < r.total_invested_b

    def test_evaluate_at_is_end_age_last_month(self):
        s = _monthly_monotonic("1980-01-31", 40 * 12 + 12)
        r = simulate_age_vs_amount(s, birth_year=1960)
        assert r.evaluate_at == pd.Timestamp("2019-12-31")
        assert r.a_start == pd.Timestamp("1980-01-31")
        assert r.a_end == pd.Timestamp("2019-12-31")
        assert r.b_start == pd.Timestamp("2000-01-31")
        assert r.b_end == pd.Timestamp("2019-12-31")


class TestAnalyticCase:
    def test_known_values_single_year(self):
        a_monthly = 1.0
        a_years = 1
        b_monthly = 2.0
        b_years = 1
        idx = pd.date_range("2020-01-31", periods=24, freq="ME")
        s = pd.Series([10.0] * 24, index=idx)

        r = simulate_age_vs_amount(
            s, birth_year=2000,
            a_start_age=20, a_monthly_amount=a_monthly, a_duration_years=a_years,
            b_start_age=20, b_monthly_amount=b_monthly, b_duration_years=b_years,
            end_age=21,
        )
        assert r.total_invested_a == pytest.approx(12.0)
        assert r.total_invested_b == pytest.approx(24.0)
        assert r.final_value_a == pytest.approx(12.0)
        assert r.final_value_b == pytest.approx(24.0)
        assert r.winner == "B"

    def test_step_up_prices_analytic(self):
        idx = pd.date_range("2020-01-31", periods=24, freq="ME")
        prices = [10.0] * 12 + [20.0] * 12
        s = pd.Series(prices, index=idx)

        r = simulate_age_vs_amount(
            s, birth_year=2000,
            a_start_age=20, a_monthly_amount=1.0, a_duration_years=1,
            b_start_age=20, b_monthly_amount=1.0, b_duration_years=1,
            end_age=22,
        )
        assert r.final_value_a == pytest.approx(1.2 * 20.0)
        assert r.final_value_b == pytest.approx(1.2 * 20.0)


class TestBatch:
    def test_batch_returns_rows_for_each_birth_year(self):
        s = _monthly_flat("1980-01-31", 40 * 12 + 12 + 12)
        df = simulate_age_vs_amount_batch(s, birth_years=[1960, 1961])
        assert set(df.index) == {1960, 1961}
        assert "final_value_a" in df.columns
        assert "winner" in df.columns

    def test_batch_skip_insufficient_by_default(self):
        s = _monthly_flat("1980-01-31", 40 * 12)
        df = simulate_age_vs_amount_batch(s, birth_years=[1960, 1961, 1962])
        assert 1960 in df.index
        assert 1961 not in df.index
        assert 1962 not in df.index

    def test_batch_can_raise_on_insufficient(self):
        s = _monthly_flat("1980-01-31", 40 * 12)
        with pytest.raises(InsufficientDataError):
            simulate_age_vs_amount_batch(
                s, birth_years=[1960, 1961], skip_insufficient=False
            )

    def test_batch_empty_returns_empty_df(self):
        s = _monthly_flat("1980-01-31", 5)
        df = simulate_age_vs_amount_batch(s, birth_years=[1960])
        assert df.empty

    def test_batch_custom_params_propagate(self):
        s = _monthly_flat("1980-01-31", 30 * 12 + 12)
        df = simulate_age_vs_amount_batch(
            s, birth_years=[1960], end_age=50, a_duration_years=30, b_duration_years=10,
            a_start_age=20, b_start_age=40,
        )
        assert not df.empty
        row = df.iloc[0]
        assert row["total_invested_a"] == pytest.approx(30_000 * 12 * 30)
        assert row["total_invested_b"] == pytest.approx(100_000 * 12 * 10)
