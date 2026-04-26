"""仮説 003 DCA Loss Horizon: 積立開始から「評価額 > 元本」になるまでの月数。

問い:
    各 start_month で月次 DCA を開始した場合、
    保有資産評価額が累積元本を **初めて** 上回るまで何ヶ月かかるか。

定義:
    - 月次 DCA: 各月の close で monthly_amount を購入 → ユニット蓄積
    - value[i]    = cumulative_units[i] * close[i]
    - invested[i] = monthly_amount * (i + 1)                  # i は 0-based
    - 解析対象指標: min { i : value[i] > invested[i] }
      * i = 0 では必ず value == invested（自明）。
      * k := 利益化までの経過月数（0-based index）。k は 1 以上でのみ意味を持つ。
    - 期間末までに達しなければ reached_profit = False を返す。

本ファイルは純粋関数。ネットワーク / ファイル I/O は一切含まない。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


class InsufficientDataError(ValueError):
    """start_month が monthly_close 範囲外など、シミュレーション不能な入力時に送出。"""


@dataclass(frozen=True)
class LossHorizonResult:
    """単一 start_month の Loss-Horizon シミュレーション結果。

    Attributes:
        start_month:            シリーズ上で採用された最初の購入月
        months_available:       start_month 以降に存在する月数（購入回数に一致）
        monthly_amount:         毎月の拠出額
        reached_profit:         期間内に value > invested が発生したか
        months_to_profit:       発生時の経過月数（0-based）。未達なら None
        years_to_profit:        months_to_profit / 12 （float）。未達なら None
        profit_month:           発生月（未達なら None）
        final_month:            データ末端月
        final_invested:         最終月時点の累積元本
        final_value:            最終月時点の評価額
        final_ratio:            final_value / final_invested
    """

    start_month: pd.Timestamp
    months_available: int
    monthly_amount: float
    reached_profit: bool
    months_to_profit: int | None
    years_to_profit: float | None
    profit_month: pd.Timestamp | None
    final_month: pd.Timestamp
    final_invested: float
    final_value: float
    final_ratio: float


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


def simulate_dca_loss_horizon(
    monthly_close: pd.Series,
    start_month: pd.Timestamp | str,
    *,
    monthly_amount: float = 30_000.0,
    profit_rel_tol: float = 1e-9,
) -> LossHorizonResult:
    """単一 start_month での DCA Loss-Horizon 計算。

    start_month は series の index に正確一致しなくてよい。
    「start_month 以降で最も早い月」が最初の購入月として採用される。

    Args:
        profit_rel_tol:
            「value > invested * (1 + profit_rel_tol)」を profitable と判定するための
            相対許容誤差。フラット市場における浮動小数の誤差を「利益」と誤検出しないため。
    """
    if monthly_amount <= 0:
        raise ValueError("monthly_amount must be strictly positive")
    if profit_rel_tol < 0:
        raise ValueError("profit_rel_tol must be >= 0")

    series = _validate_series(monthly_close)
    start_ts = pd.Timestamp(start_month)

    sub = series.loc[start_ts:]
    if sub.empty:
        raise InsufficientDataError(
            f"start_month={start_ts.date()} is after the last available month "
            f"{series.index.max().date()}"
        )

    closes = sub.to_numpy(dtype="float64")
    n = closes.size

    cumulative_units = np.cumsum(monthly_amount / closes)
    value = cumulative_units * closes
    invested = monthly_amount * np.arange(1, n + 1, dtype="float64")

    profitable_mask = value > invested * (1.0 + profit_rel_tol)
    if profitable_mask.any():
        first_idx = int(np.argmax(profitable_mask))
        reached = True
        months_to_profit = first_idx
        years_to_profit = first_idx / 12.0
        profit_month = sub.index[first_idx]
    else:
        reached = False
        months_to_profit = None
        years_to_profit = None
        profit_month = None

    final_month = sub.index[-1]
    final_invested = float(invested[-1])
    final_value = float(value[-1])
    final_ratio = final_value / final_invested if final_invested > 0 else float("nan")

    return LossHorizonResult(
        start_month=sub.index[0],
        months_available=n,
        monthly_amount=float(monthly_amount),
        reached_profit=reached,
        months_to_profit=months_to_profit,
        years_to_profit=years_to_profit,
        profit_month=profit_month,
        final_month=final_month,
        final_invested=final_invested,
        final_value=final_value,
        final_ratio=final_ratio,
    )


def simulate_dca_loss_horizon_batch(
    monthly_close: pd.Series,
    *,
    monthly_amount: float = 30_000.0,
    min_months_available: int = 1,
    profit_rel_tol: float = 1e-9,
) -> pd.DataFrame:
    """シリーズの各月を start_month として一括評価し、DataFrame を返す。

    Args:
        min_months_available: 最小必要残月数。残り月数がこれ未満の start_month は除外。
                              Loss-Horizon 分析では期間末付近での誤った "reached=False"
                              を排除する目的で値を大きめに取ることが多い（例: 12）。

    Returns:
        DataFrame (index=start_month)。カラムは LossHorizonResult のフィールド。
    """
    series = _validate_series(monthly_close)
    if min_months_available < 1:
        raise ValueError("min_months_available must be >= 1")

    rows: list[dict] = []
    n_total = len(series)
    for i in range(n_total):
        months_available = n_total - i
        if months_available < min_months_available:
            break
        r = simulate_dca_loss_horizon(
            series.iloc[i:],
            series.index[i],
            monthly_amount=monthly_amount,
            profit_rel_tol=profit_rel_tol,
        )
        rows.append(asdict(r))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("start_month")
