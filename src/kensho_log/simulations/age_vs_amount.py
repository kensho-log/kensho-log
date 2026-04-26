"""仮説 002 Age vs Amount: 20歳月3万 vs 40歳月10万 を 60歳時点で比較。

条件（デフォルト）:
    A. 開始年齢 20 / 毎月 30,000 / 40年間拠出
    B. 開始年齢 40 / 毎月 100,000 / 20年間拠出
    評価時点: 年齢 60 の末月

拠出期間終了後は「保持（holding）」状態。条件 B は拠出期間の最終年と評価年が
等しくなるが、拠出後の値動きを反映するため評価は eval_year の最終月 close で行う。

入力:
    monthly_close: pandas.Series (index=DatetimeIndex, 月末 close)
    birth_year:    出生年（int）

出力:
    AgeVsAmountResult dataclass（単発）または pandas.DataFrame（バッチ）

本ファイルは純粋関数。ネットワーク / ファイル I/O は一切含まない。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import pandas as pd


class InsufficientDataError(ValueError):
    """シミュレーションに必要な期間が monthly_close に含まれていない場合に送出。"""


@dataclass(frozen=True)
class AgeVsAmountResult:
    """単一出生年の Age-vs-Amount シミュレーション結果。"""

    birth_year: int
    a_start: pd.Timestamp
    a_end: pd.Timestamp
    b_start: pd.Timestamp
    b_end: pd.Timestamp
    evaluate_at: pd.Timestamp
    final_value_a: float
    final_value_b: float
    total_invested_a: float
    total_invested_b: float
    winner: str
    diff_pct_a_vs_b: float


def _validate_series(series: pd.Series) -> pd.Series:
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


def _slice_years(series: pd.Series, first_year: int, last_year: int) -> pd.Series:
    """[first_year-01-01, last_year-12-31] を両端含む形でスライス。"""
    start = pd.Timestamp(f"{first_year}-01-01")
    end = pd.Timestamp(f"{last_year}-12-31")
    return series.loc[start:end]


def _dca_and_value(
    monthly_close: pd.Series,
    purchase: pd.Series,
    evaluate_at: pd.Timestamp,
    monthly_amount: float,
) -> tuple[float, float]:
    units = float((monthly_amount / purchase).sum())
    final_close = float(monthly_close.loc[evaluate_at])
    return units * final_close, float(monthly_amount * len(purchase))


def simulate_age_vs_amount(
    monthly_close: pd.Series,
    birth_year: int,
    *,
    a_start_age: int = 20,
    a_monthly_amount: float = 30_000.0,
    a_duration_years: int = 40,
    b_start_age: int = 40,
    b_monthly_amount: float = 100_000.0,
    b_duration_years: int = 20,
    end_age: int = 60,
) -> AgeVsAmountResult:
    """単一 birth_year に対し、条件A/B を同一評価時点で計算する。

    Raises:
        InsufficientDataError: 必要年の一部が monthly_close に欠ける場合。
        ValueError: パラメータ整合性エラー（年齢の逆転、期間超過など）。
    """
    if a_monthly_amount <= 0 or b_monthly_amount <= 0:
        raise ValueError("monthly amounts must be strictly positive")
    if a_duration_years <= 0 or b_duration_years <= 0:
        raise ValueError("durations must be strictly positive")
    if a_start_age >= end_age or b_start_age >= end_age:
        raise ValueError("start_age must be strictly less than end_age")
    if a_start_age + a_duration_years > end_age:
        raise ValueError(
            "a_start_age + a_duration_years must be <= end_age "
            f"(got {a_start_age}+{a_duration_years} > {end_age})"
        )
    if b_start_age + b_duration_years > end_age:
        raise ValueError(
            "b_start_age + b_duration_years must be <= end_age "
            f"(got {b_start_age}+{b_duration_years} > {end_age})"
        )

    series = _validate_series(monthly_close)

    a_first = birth_year + a_start_age
    a_last = birth_year + a_start_age + a_duration_years - 1
    b_first = birth_year + b_start_age
    b_last = birth_year + b_start_age + b_duration_years - 1
    eval_year = birth_year + end_age - 1

    year_counts = series.index.year.value_counts()

    def _require_full_year(y: int, label: str) -> None:
        cnt = int(year_counts.get(y, 0))
        if cnt < 12:
            raise InsufficientDataError(
                f"{label}: year {y} has {cnt} monthly rows in monthly_close (need 12)"
            )

    for y in range(a_first, a_last + 1):
        _require_full_year(y, "condition A purchase")
    for y in range(b_first, b_last + 1):
        _require_full_year(y, "condition B purchase")
    eval_cnt = int(year_counts.get(eval_year, 0))
    if eval_cnt == 0:
        raise InsufficientDataError(
            f"evaluation year {eval_year} has no rows in monthly_close"
        )

    purchase_a = _slice_years(series, a_first, a_last)
    purchase_b = _slice_years(series, b_first, b_last)
    eval_slice = _slice_years(series, eval_year, eval_year)

    evaluate_at = eval_slice.index[-1]
    a_start_ts = purchase_a.index[0]
    a_end_ts = purchase_a.index[-1]
    b_start_ts = purchase_b.index[0]
    b_end_ts = purchase_b.index[-1]

    final_a, inv_a = _dca_and_value(series, purchase_a, evaluate_at, a_monthly_amount)
    final_b, inv_b = _dca_and_value(series, purchase_b, evaluate_at, b_monthly_amount)

    if final_b == 0:
        diff_pct = float("nan")
    else:
        diff_pct = (final_a - final_b) / final_b * 100.0

    if final_a > final_b:
        winner = "A"
    elif final_a < final_b:
        winner = "B"
    else:
        winner = "TIE"

    return AgeVsAmountResult(
        birth_year=birth_year,
        a_start=a_start_ts,
        a_end=a_end_ts,
        b_start=b_start_ts,
        b_end=b_end_ts,
        evaluate_at=evaluate_at,
        final_value_a=final_a,
        final_value_b=final_b,
        total_invested_a=inv_a,
        total_invested_b=inv_b,
        winner=winner,
        diff_pct_a_vs_b=diff_pct,
    )


def simulate_age_vs_amount_batch(
    monthly_close: pd.Series,
    birth_years: Iterable[int],
    *,
    skip_insufficient: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """複数 birth_year を一括評価し、DataFrame（index=birth_year）を返す。

    Args:
        skip_insufficient:
            True: データ不足の birth_year を結果から除外
            False: 最初に出会った不足で InsufficientDataError を送出
    """
    rows: list[dict] = []
    for y in birth_years:
        try:
            r = simulate_age_vs_amount(monthly_close, y, **kwargs)
        except InsufficientDataError:
            if skip_insufficient:
                continue
            raise
        rows.append(asdict(r))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("birth_year")
