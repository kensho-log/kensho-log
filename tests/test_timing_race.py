"""simulate_timing_race / summarize_timing_race の単体テスト。

- 純粋関数なのでネットワーク不要
- 解析的に結果が求まるケースで厳密検証
- 戦略間の順序不変条件（A >= B >= C）を不変量として検証
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kensho_log.simulations.timing_race import (
    TimingRaceSummary,
    simulate_timing_race,
    summarize_timing_race,
)


def _monthly_index(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=periods, freq="ME")


def _flat_series(start: str, periods: int, price: float = 100.0) -> pd.Series:
    idx = _monthly_index(start, periods)
    return pd.Series([price] * periods, index=idx, name="close")


def _linear_series(
    start: str, periods: int, base: float = 100.0, step: float = 1.0
) -> pd.Series:
    idx = _monthly_index(start, periods)
    vals = [base + step * i for i in range(periods)]
    return pd.Series(vals, index=idx, name="close")


class TestValidation:
    def test_non_series_raises(self):
        with pytest.raises(TypeError):
            simulate_timing_race([1.0, 2.0])  # type: ignore[arg-type]

    def test_non_datetime_index_raises(self):
        s = pd.Series([1.0, 2.0], index=[0, 1])
        with pytest.raises(TypeError):
            simulate_timing_race(s)

    def test_empty_raises(self):
        s = pd.Series([], index=pd.DatetimeIndex([]), dtype="float64")
        with pytest.raises(ValueError):
            simulate_timing_race(s)

    def test_duplicate_index_raises(self):
        idx = pd.DatetimeIndex(["2020-01-31", "2020-01-31"])
        s = pd.Series([100.0, 100.0], index=idx)
        with pytest.raises(ValueError):
            simulate_timing_race(s)

    def test_negative_price_raises(self):
        s = pd.Series([100.0, -1.0], index=_monthly_index("2020-01-31", 2))
        with pytest.raises(ValueError):
            simulate_timing_race(s)

    def test_nan_raises(self):
        s = pd.Series([100.0, np.nan], index=_monthly_index("2020-01-31", 2))
        with pytest.raises(ValueError):
            simulate_timing_race(s)

    def test_nonpositive_budget_raises(self):
        s = _flat_series("2020-01-31", 12)
        with pytest.raises(ValueError):
            simulate_timing_race(s, annual_budget=0.0)
        with pytest.raises(ValueError):
            simulate_timing_race(s, annual_budget=-1.0)

    def test_unsorted_index_is_sorted_internally(self):
        s_sorted = _linear_series("2020-01-31", 12, base=100.0, step=1.0)
        shuffled = s_sorted.iloc[[5, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]]
        out1 = simulate_timing_race(s_sorted, annual_budget=1_200_000)
        out2 = simulate_timing_race(shuffled, annual_budget=1_200_000)
        pd.testing.assert_index_equal(out1.index, out2.index, check_exact=True, exact="equiv")
        pd.testing.assert_frame_equal(
            out1.reset_index(drop=True),
            out2.reset_index(drop=True),
        )


class TestFlatMarket:
    def test_flat_market_all_equal_to_invested(self):
        s = _flat_series("2020-01-31", 12, price=100.0)
        out = simulate_timing_race(s, annual_budget=1_200_000)
        last = out.iloc[-1]
        assert last["A_value"] == pytest.approx(1_200_000.0)
        assert last["B_value"] == pytest.approx(1_200_000.0)
        assert last["C_value"] == pytest.approx(1_200_000.0)
        assert last["A_invested_cum"] == pytest.approx(1_200_000.0)
        assert last["B_invested_cum"] == pytest.approx(1_200_000.0)
        assert last["C_invested_cum"] == pytest.approx(1_200_000.0)

    def test_flat_market_summary_zero_diff(self):
        s = _flat_series("2020-01-31", 12, price=100.0)
        out = simulate_timing_race(s, annual_budget=1_200_000)
        summary = summarize_timing_race(out)
        assert isinstance(summary, TimingRaceSummary)
        assert summary.diff_pct_a_vs_c == pytest.approx(0.0, abs=1e-9)
        assert summary.diff_pct_b_vs_c == pytest.approx(0.0, abs=1e-9)
        assert summary.years == 1


class TestMonotonicUp:
    def test_monotonic_up_one_year_invariants(self):
        s = _linear_series("2020-01-31", 12, base=100.0, step=10.0)
        out = simulate_timing_race(s, annual_budget=1_200_000)
        last = out.iloc[-1]

        assert last["A_value"] > last["B_value"] > last["C_value"]
        assert last["A_units_cum"] > last["B_units_cum"] > last["C_units_cum"]
        assert last["A_invested_cum"] == pytest.approx(last["C_invested_cum"])
        assert last["A_invested_cum"] == pytest.approx(last["B_invested_cum"])

    def test_monotonic_up_A_buys_jan_C_buys_dec(self):
        s = _linear_series("2020-01-31", 12, base=100.0, step=10.0)
        out = simulate_timing_race(s, annual_budget=1_200_000)

        assert out["A_invested_cum"].iloc[0] == pytest.approx(1_200_000.0)
        assert out["A_invested_cum"].iloc[-1] == pytest.approx(1_200_000.0)
        assert out["C_invested_cum"].iloc[-1] == pytest.approx(1_200_000.0)
        assert out["C_invested_cum"].iloc[-2] == pytest.approx(0.0)

    def test_monotonic_up_analytic_A_value(self):
        s = _linear_series("2020-01-31", 12, base=100.0, step=10.0)
        out = simulate_timing_race(s, annual_budget=1_200_000)
        last = out.iloc[-1]
        expected_units_a = 1_200_000.0 / 100.0
        expected_value_a = expected_units_a * 210.0
        assert last["A_units_cum"] == pytest.approx(expected_units_a)
        assert last["A_value"] == pytest.approx(expected_value_a)


class TestMonotonicDown:
    def test_monotonic_down_invariants(self):
        s = _linear_series("2020-01-31", 12, base=210.0, step=-10.0)
        out = simulate_timing_race(s, annual_budget=1_200_000)
        last = out.iloc[-1]

        assert last["A_units_cum"] > last["B_units_cum"] > last["C_units_cum"]
        assert last["A_value"] > last["B_value"] > last["C_value"]
        assert last["A_invested_cum"] == pytest.approx(last["C_invested_cum"])


class TestMultiYear:
    def test_two_years_full_invariants(self):
        idx = _monthly_index("2020-01-31", 24)
        vals = [float(100 + (i % 12) * 10) for i in range(24)]
        s = pd.Series(vals, index=idx)

        out = simulate_timing_race(s, annual_budget=1_000_000)
        last = out.iloc[-1]
        assert last["A_invested_cum"] == pytest.approx(2_000_000.0)
        assert last["B_invested_cum"] == pytest.approx(2_000_000.0)
        assert last["C_invested_cum"] == pytest.approx(2_000_000.0)
        assert last["A_value"] > last["B_value"] > last["C_value"]

        summary = summarize_timing_race(out)
        assert summary.years == 2
        assert summary.period_start == idx[0]
        assert summary.period_end == idx[-1]
        assert summary.diff_pct_a_vs_c > 0.0
        assert summary.diff_pct_b_vs_c > 0.0
        assert summary.diff_pct_a_vs_c > summary.diff_pct_b_vs_c

    def test_invested_per_year_is_budget(self):
        idx = _monthly_index("2020-01-31", 36)
        rng = np.random.default_rng(seed=42)
        vals = 100.0 + np.cumsum(rng.normal(0, 1, size=36)).clip(-50)
        vals = np.where(vals <= 0, 1.0, vals)
        s = pd.Series(vals, index=idx)

        out = simulate_timing_race(s, annual_budget=1_000_000)
        for year in (2020, 2021, 2022):
            year_mask = out["year"] == year
            a_this_year = (
                out.loc[year_mask, "A_invested_cum"].iloc[-1]
                - (
                    out.loc[~year_mask & (out["year"] < year), "A_invested_cum"].iloc[-1]
                    if (out["year"] < year).any()
                    else 0.0
                )
            )
            c_this_year = (
                out.loc[year_mask, "C_invested_cum"].iloc[-1]
                - (
                    out.loc[~year_mask & (out["year"] < year), "C_invested_cum"].iloc[-1]
                    if (out["year"] < year).any()
                    else 0.0
                )
            )
            assert a_this_year == pytest.approx(1_000_000.0)
            assert c_this_year == pytest.approx(1_000_000.0)


class TestSummaryMissingColumns:
    def test_missing_columns_raise(self):
        df = pd.DataFrame({"close": [100.0]}, index=_monthly_index("2020-01-31", 1))
        with pytest.raises(ValueError):
            summarize_timing_race(df)

    def test_empty_result_raises(self):
        cols = {
            "A_value": [], "B_value": [], "C_value": [],
            "A_invested_cum": [], "B_invested_cum": [], "C_invested_cum": [],
            "year": [],
        }
        df = pd.DataFrame(cols, index=pd.DatetimeIndex([]))
        with pytest.raises(ValueError):
            summarize_timing_race(df)
