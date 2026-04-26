"""データ取得・二重化検証レイヤ。

外部 API (yfinance / Stooq) からのデータ取得はすべてこのパッケージを経由する。
検証スクリプト側は取得方法や二重化ロジックを意識しない。
"""

from .sp500 import (
    CrossCheckResult,
    SP500DataError,
    fetch_sp500,
    fetch_sp500_monthly,
)

__all__ = [
    "CrossCheckResult",
    "SP500DataError",
    "fetch_sp500",
    "fetch_sp500_monthly",
]
