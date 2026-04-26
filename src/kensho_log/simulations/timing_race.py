"""仮説 001 Timing Race: 完璧タイミング / ドルコスト / 最悪タイミング 3者レース。

戦略定義（全戦略で年間投資額は同額）:
    A. Perfect Timing (in hindsight):
        各暦年内で close が最小の月に 年間予算の全額を一括投資。
    B. Dollar-Cost Averaging:
        各月に 年間予算 / 12 を一括投資。
    C. Worst Timing (in hindsight):
        各暦年内で close が最大の月に 年間予算の全額を一括投資。

投資額の整合性:
    入力期間が完全な暦年（各年 12 ヶ月揃い）の場合、A / B / C の総投資額は
    期間年数 × annual_budget で一致する。
    不完全な年の扱いは呼び出し側の責務とし、本関数は
    「存在する月から年ごとの min/max を選ぶ」「B は存在する月すべてに等額」
    のルールを機械的に適用する。

入力:
    monthly_close: pandas.Series (index=DatetimeIndex, values=close price, 月末)
入力前提:
    - index は昇順かつ重複なし
    - values は正の値のみ（0 / 負は ValueError）
    - 配当再投資込み Total Return を想定（呼び出し側で ^SP500TR を渡す）

出力:
    pandas.DataFrame
        index: monthly_close.index
        columns:
            close, year
            A_invested_cum, B_invested_cum, C_invested_cum
            A_units_cum, B_units_cum, C_units_cum
            A_value, B_value, C_value

本ファイルは純粋関数のみ。ネットワーク・ファイル I/O・plot は一切含まない。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimingRaceSummary:
    """Timing Race の最終サマリ。

    Attributes:
        final_value_a: 最終月時点の A 戦略評価額
        final_value_b: 最終月時点の B 戦略評価額
        final_value_c: 最終月時点の C 戦略評価額
        total_invested_a: A の累積投資額
        total_invested_b: B の累積投資額
        total_invested_c: C の累積投資額
        diff_pct_a_vs_c: (A - C) / C * 100 最終値（A の優位分 %）
        diff_pct_b_vs_c: (B - C) / C * 100
        period_start: シミュレーション開始月
        period_end:   シミュレーション終了月
        years: 含まれる暦年数（distinct）
    """

    final_value_a: float
    final_value_b: float
    final_value_c: float
    total_invested_a: float
    total_invested_b: float
    total_invested_c: float
    diff_pct_a_vs_c: float
    diff_pct_b_vs_c: float
    period_start: pd.Timestamp
    period_end: pd.Timestamp
    years: int


def _validate_input(series: pd.Series) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError("monthly_close must be a pandas.Series")
    if series.empty:
        raise ValueError("monthly_close must not be empty")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("monthly_close.index must be a pandas.DatetimeIndex")
    if series.index.has_duplicates:
        raise ValueError("monthly_close.index must not contain duplicates")
    if not series.index.is_monotonic_increasing:
        series = series.sort_index()
    if series.isna().any():
        raise ValueError("monthly_close must not contain NaN")
    if (series <= 0).any():
        raise ValueError("monthly_close must be strictly positive")
    return series.astype("float64")


def simulate_timing_race(
    monthly_close: pd.Series,
    annual_budget: float = 1_000_000.0,
) -> pd.DataFrame:
    """3戦略の累積投資額・保有ユニット数・評価額を月次で計算する。

    Args:
        monthly_close: 月末 close の時系列（index=DatetimeIndex）
        annual_budget: 1 暦年あたりの投資予算（全戦略で同額）

    Returns:
        pandas.DataFrame: 入力 index と同じ行を持つ DataFrame。
        列は close, year, *_invested_cum, *_units_cum, *_value。
    """
    if annual_budget <= 0:
        raise ValueError("annual_budget must be strictly positive")

    close = _validate_input(monthly_close).rename("close")
    years = close.index.year
    df = pd.DataFrame({"close": close.values, "year": years}, index=close.index)

    invest_a = pd.Series(0.0, index=close.index)
    invest_c = pd.Series(0.0, index=close.index)

    for year, g in df.groupby("year", sort=True):
        min_idx = g["close"].idxmin()
        max_idx = g["close"].idxmax()
        invest_a.loc[min_idx] += float(annual_budget)
        invest_c.loc[max_idx] += float(annual_budget)

    invest_b = pd.Series(
        float(annual_budget) / 12.0, index=close.index, dtype="float64"
    )

    units_a_step = invest_a / df["close"]
    units_b_step = invest_b / df["close"]
    units_c_step = invest_c / df["close"]

    result = pd.DataFrame(
        {
            "close": df["close"],
            "year": df["year"].astype("int64"),
            "A_invested_cum": invest_a.cumsum(),
            "B_invested_cum": invest_b.cumsum(),
            "C_invested_cum": invest_c.cumsum(),
            "A_units_cum": units_a_step.cumsum(),
            "B_units_cum": units_b_step.cumsum(),
            "C_units_cum": units_c_step.cumsum(),
        },
        index=close.index,
    )
    result["A_value"] = result["A_units_cum"] * result["close"]
    result["B_value"] = result["B_units_cum"] * result["close"]
    result["C_value"] = result["C_units_cum"] * result["close"]

    return result


def summarize_timing_race(result: pd.DataFrame) -> TimingRaceSummary:
    """simulate_timing_race の出力から最終サマリを計算する。"""
    required = {
        "A_value", "B_value", "C_value",
        "A_invested_cum", "B_invested_cum", "C_invested_cum",
        "year",
    }
    missing = required - set(result.columns)
    if missing:
        raise ValueError(f"result is missing columns: {sorted(missing)}")
    if result.empty:
        raise ValueError("result must not be empty")

    last = result.iloc[-1]
    fa = float(last["A_value"])
    fb = float(last["B_value"])
    fc = float(last["C_value"])

    if fc == 0 or np.isclose(fc, 0.0):
        diff_a = float("nan")
        diff_b = float("nan")
    else:
        diff_a = (fa - fc) / fc * 100.0
        diff_b = (fb - fc) / fc * 100.0

    return TimingRaceSummary(
        final_value_a=fa,
        final_value_b=fb,
        final_value_c=fc,
        total_invested_a=float(last["A_invested_cum"]),
        total_invested_b=float(last["B_invested_cum"]),
        total_invested_c=float(last["C_invested_cum"]),
        diff_pct_a_vs_c=diff_a,
        diff_pct_b_vs_c=diff_b,
        period_start=result.index[0],
        period_end=result.index[-1],
        years=int(result["year"].nunique()),
    )
