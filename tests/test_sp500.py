"""src/kensho_log/data_sources/sp500.py の単体テスト。

ネットワーク・外部 API には依存しない。
下位の _fetch_yfinance / _fetch_stooq をモック差し替えして検証する。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kensho_log.data_sources import sp500


def _make_df(start: str, periods: int, base: float = 1000.0, step: float = 1.0) -> pd.DataFrame:
    idx = pd.date_range(start=start, periods=periods, freq="B")
    close = np.array([base + step * i for i in range(periods)], dtype=np.float32)
    return pd.DataFrame({"close": close}, index=idx).rename_axis("date")


def _patch_fetchers(monkeypatch, yf_map: dict[str, pd.DataFrame], stooq_map: dict[str, pd.DataFrame]):
    def fake_yf(ticker, start, end):
        if ticker not in yf_map:
            raise sp500.SP500DataError(f"no fixture for {ticker}")
        return yf_map[ticker].copy()

    def fake_stooq(ticker, start, end):
        if ticker not in stooq_map:
            raise sp500.SP500DataError(f"no fixture for {ticker}")
        return stooq_map[ticker].copy()

    monkeypatch.setattr(sp500, "_fetch_yfinance", fake_yf)
    monkeypatch.setattr(sp500, "_fetch_stooq", fake_stooq)


class TestCrossCheck:
    def test_identical_series_passes(self):
        a = _make_df("2020-01-01", 10)
        result = sp500._cross_check(a, a.copy(), "src_a", "src_b", tolerance_pct=0.1)
        assert result.passed is True
        assert result.max_diff_pct == pytest.approx(0.0, abs=1e-6)
        assert result.mean_diff_pct == pytest.approx(0.0, abs=1e-6)

    def test_small_diff_within_tolerance(self):
        a = _make_df("2020-01-01", 10, base=1000.0)
        b = a.copy()
        b["close"] = b["close"] * 1.0005
        result = sp500._cross_check(a, b, "src_a", "src_b", tolerance_pct=0.1)
        assert result.passed is True
        assert result.max_diff_pct < 0.1

    def test_large_diff_fails(self):
        a = _make_df("2020-01-01", 10, base=1000.0)
        b = a.copy()
        b["close"] = b["close"] * 1.05
        result = sp500._cross_check(a, b, "src_a", "src_b", tolerance_pct=0.1)
        assert result.passed is False
        assert result.max_diff_pct > 0.1

    def test_no_overlap_returns_not_passed(self):
        a = _make_df("2020-01-01", 5)
        b = _make_df("2025-01-01", 5)
        result = sp500._cross_check(a, b, "src_a", "src_b")
        assert result.passed is False
        assert np.isnan(result.max_diff_pct)
        assert result.overlap_start is None

    def test_missing_close_column_raises(self):
        a = _make_df("2020-01-01", 5).rename(columns={"close": "value"})
        b = _make_df("2020-01-01", 5)
        with pytest.raises(ValueError):
            sp500._cross_check(a, b, "src_a", "src_b")


class TestCache:
    def test_cache_write_then_read(self, tmp_path):
        df = _make_df("2020-01-01", 3)
        path = tmp_path / "x.parquet"
        sp500._write_cache(path, df)
        loaded = sp500._read_cache(path)
        assert loaded is not None
        pd.testing.assert_frame_equal(
            loaded.astype("float32"), df.astype("float32"), check_freq=False
        )

    def test_read_missing_returns_none(self, tmp_path):
        assert sp500._read_cache(tmp_path / "nope.parquet") is None


class TestFetchSP500:
    def test_total_return_returns_tr_close_and_runs_crosscheck(self, monkeypatch, tmp_path):
        price = _make_df("2020-01-01", 5, base=3000.0)
        tr = _make_df("2020-01-01", 5, base=5000.0)
        _patch_fetchers(
            monkeypatch,
            yf_map={
                sp500.TICKER_TOTAL_RETURN: tr,
                sp500.TICKER_PRICE_YF: price,
            },
            stooq_map={sp500.TICKER_PRICE_STOOQ: price.copy()},
        )

        df, cross = sp500.fetch_sp500(
            series="total_return",
            start="2020-01-01",
            end="2020-01-31",
            cache_dir=tmp_path,
            refresh=True,
        )

        assert list(df.columns) == ["close"]
        assert df["close"].iloc[0] == pytest.approx(5000.0, abs=1e-3)
        assert cross.passed is True
        assert cross.max_diff_pct == pytest.approx(0.0, abs=1e-6)
        assert cross.source_a.startswith("yfinance:")
        assert cross.source_b.startswith("stooq:")

    def test_price_series_returns_price(self, monkeypatch, tmp_path):
        price = _make_df("2020-01-01", 5, base=3000.0)
        _patch_fetchers(
            monkeypatch,
            yf_map={
                sp500.TICKER_PRICE_YF: price,
                sp500.TICKER_TOTAL_RETURN: price,
            },
            stooq_map={sp500.TICKER_PRICE_STOOQ: price.copy()},
        )

        df, cross = sp500.fetch_sp500(
            series="price",
            start="2020-01-01",
            end="2020-01-31",
            cache_dir=tmp_path,
            refresh=True,
        )
        assert df["close"].iloc[0] == pytest.approx(3000.0, abs=1e-3)
        assert cross.passed is True

    def test_crosscheck_fails_but_does_not_raise(self, monkeypatch, tmp_path, caplog):
        price_yf = _make_df("2020-01-01", 5, base=3000.0)
        price_stooq = price_yf.copy()
        price_stooq["close"] = price_stooq["close"] * 1.02
        _patch_fetchers(
            monkeypatch,
            yf_map={
                sp500.TICKER_PRICE_YF: price_yf,
                sp500.TICKER_TOTAL_RETURN: price_yf,
            },
            stooq_map={sp500.TICKER_PRICE_STOOQ: price_stooq},
        )

        _, cross = sp500.fetch_sp500(
            series="total_return",
            cache_dir=tmp_path,
            refresh=True,
        )
        assert cross.passed is False
        assert cross.max_diff_pct > 0.1

    def test_cache_hit_avoids_refetch(self, monkeypatch, tmp_path):
        price = _make_df("2020-01-01", 5, base=3000.0)
        calls = {"yf": 0, "stooq": 0}

        def fake_yf(ticker, start, end):
            calls["yf"] += 1
            if ticker == sp500.TICKER_PRICE_YF:
                return price.copy()
            if ticker == sp500.TICKER_TOTAL_RETURN:
                return price.copy()
            raise sp500.SP500DataError(ticker)

        def fake_stooq(ticker, start, end):
            calls["stooq"] += 1
            return price.copy()

        monkeypatch.setattr(sp500, "_fetch_yfinance", fake_yf)
        monkeypatch.setattr(sp500, "_fetch_stooq", fake_stooq)

        sp500.fetch_sp500(series="total_return", cache_dir=tmp_path, refresh=True)
        first_yf, first_stooq = calls["yf"], calls["stooq"]
        assert first_yf >= 2
        assert first_stooq >= 1

        sp500.fetch_sp500(series="total_return", cache_dir=tmp_path, refresh=False)
        assert calls["yf"] == first_yf
        assert calls["stooq"] == first_stooq

    def test_invalid_series_raises(self, monkeypatch, tmp_path):
        price = _make_df("2020-01-01", 5)
        _patch_fetchers(
            monkeypatch,
            yf_map={
                sp500.TICKER_PRICE_YF: price,
                sp500.TICKER_TOTAL_RETURN: price,
            },
            stooq_map={sp500.TICKER_PRICE_STOOQ: price},
        )
        with pytest.raises(ValueError):
            sp500.fetch_sp500(series="garbage", cache_dir=tmp_path, refresh=True)  # type: ignore[arg-type]


class TestMonthly:
    def test_monthly_resample_length_and_values(self, monkeypatch, tmp_path):
        idx = pd.date_range("2020-01-01", "2020-03-31", freq="B")
        close = np.linspace(1000.0, 1100.0, num=len(idx), dtype=np.float32)
        tr = pd.DataFrame({"close": close}, index=idx).rename_axis("date")
        price = tr.copy()
        _patch_fetchers(
            monkeypatch,
            yf_map={
                sp500.TICKER_TOTAL_RETURN: tr,
                sp500.TICKER_PRICE_YF: price,
            },
            stooq_map={sp500.TICKER_PRICE_STOOQ: price.copy()},
        )

        monthly, cross = sp500.fetch_sp500_monthly(
            series="total_return", cache_dir=tmp_path, refresh=True
        )
        assert len(monthly) == 3
        assert monthly.index.name == "month_end"
        assert list(monthly.columns) == ["close"]
        assert monthly["close"].iloc[-1] == pytest.approx(float(close[-1]), abs=1e-3)
        assert cross.passed is True


class TestDefaultCacheDir:
    def test_default_cache_dir_under_repo_data_raw(self):
        assert sp500.DEFAULT_CACHE_DIR.parts[-3:] == ("data", "raw", "sp500")
        assert sp500.DEFAULT_CACHE_DIR.is_absolute()
