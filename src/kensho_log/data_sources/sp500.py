"""S&P 500 データ取得層（yfinance + Stooq の二重化）。

設計方針:
- 一次: yfinance `^SP500TR`（配当再投資込み Total Return）
- 価格系クロスチェック: yfinance `^GSPC` + Stooq `^SPX`
- ローカル pickle キャッシュ（data/raw/ 配下、pandas 標準のみで完結）
- 差異 > TOLERANCE_PCT で警告（例外は呼び出し側の判断で）
- 8GB RAM 制約のため dtype=float32、不要列は早期 drop

このモジュールはネットワーク接続がある環境で実行することを想定する。
単体テストでは `fetch_sp500` が要求する下位関数をモックしてオフライン検証する。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "raw" / "sp500"

TICKER_TOTAL_RETURN = "^SP500TR"
TICKER_PRICE_YF = "^GSPC"
TICKER_PRICE_STOOQ = "^SPX"

TOLERANCE_PCT = 0.1

Series = Literal["total_return", "price"]


class SP500DataError(RuntimeError):
    """S&P 500 データ取得に失敗した場合に送出される例外。"""


class StooqApikeyRequired(SP500DataError):
    """Stooq が apikey を要求している（2026 以降のポリシー変更）場合に送出。

    呼び出し側では二重化クロスチェックの graceful degradation を行う合図として扱う。
    環境変数 STOOQ_APIKEY を設定すれば回避可能。
    """


@dataclass(frozen=True)
class CrossCheckResult:
    """二重化検証の結果。

    Attributes:
        source_a: 一次ソース識別子（例: "yfinance:^GSPC"）
        source_b: 照合ソース識別子（例: "stooq:^SPX"）
        max_diff_pct: 重複期間における終値の最大乖離率 (%)
        mean_diff_pct: 重複期間における終値の平均乖離率 (%)
        overlap_start: 重複期間の開始日
        overlap_end: 重複期間の終了日
        passed: |max_diff_pct| <= tolerance_pct なら True
        tolerance_pct: 許容差分 (%)
    """

    source_a: str
    source_b: str
    max_diff_pct: float
    mean_diff_pct: float
    overlap_start: pd.Timestamp | None
    overlap_end: pd.Timestamp | None
    passed: bool
    tolerance_pct: float


def _ensure_cache_dir(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _cache_path(cache_dir: Path, ticker: str) -> Path:
    safe = ticker.replace("^", "").replace("/", "_")
    return cache_dir / f"{safe}.pkl"


def _read_cache(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_pickle(path)
    except Exception as exc:
        logger.warning("cache read failed (%s): %s", path, exc)
        return None


def _write_cache(path: Path, df: pd.DataFrame) -> None:
    try:
        df.to_pickle(path)
    except Exception as exc:
        logger.warning("cache write failed (%s): %s", path, exc)


def _fetch_yfinance(
    ticker: str,
    start: str | date | None,
    end: str | date | None,
) -> pd.DataFrame:
    """yfinance から単一ティッカーの日次データを取得。

    依存を遅延 import して、pytest 時に yfinance 不要化を許容。
    """
    import yfinance as yf

    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        raise SP500DataError(f"yfinance returned empty DataFrame for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    return df[["Close"]].astype(np.float32).rename(columns={"Close": "close"})


def _fmt_stooq_date(d: str | date | None) -> str | None:
    if d is None:
        return None
    if isinstance(d, str):
        d = pd.to_datetime(d).date()
    return d.strftime("%Y%m%d")


def _fetch_stooq(
    ticker: str,
    start: str | date | None,
    end: str | date | None,
) -> pd.DataFrame:
    """Stooq から日次データを取得（直接 CSV エンドポイント）。

    2026 年以降 Stooq は CSV ダウンロードに apikey を要求するようになった。
    環境変数 ``STOOQ_APIKEY`` が設定されていればクエリに付加する。
    apikey が無く、Stooq が apikey 要求レスポンスを返した場合は
    ``StooqApikeyRequired`` を送出する（呼び出し側で degradation 判断）。

    pandas_datareader 0.10 は Python 3.12 で distutils 依存の問題と
    Stooq レスポンス変化への脆弱性があるため、公開 CSV エンドポイントを
    直接叩く軽量実装に統一する。
    """
    import os
    from urllib.request import Request, urlopen

    params = {"s": ticker.lower(), "i": "d"}
    d1 = _fmt_stooq_date(start)
    d2 = _fmt_stooq_date(end)
    if d1:
        params["d1"] = d1
    if d2:
        params["d2"] = d2
    apikey = os.environ.get("STOOQ_APIKEY", "").strip()
    if apikey:
        params["apikey"] = apikey
    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"https://stooq.com/q/d/l/?{query}"

    try:
        req = Request(url, headers={"User-Agent": "kensho-log/0.1 (+https://github.com/kensho-log)"})
        with urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        raise SP500DataError(f"stooq fetch failed for {ticker}: {exc}") from exc

    head = body.lstrip()[:400].lower()
    if "apikey" in head and "," not in head.splitlines()[0]:
        raise StooqApikeyRequired(
            f"stooq requires apikey for {ticker} (set STOOQ_APIKEY env var). "
            f"First 120 chars of response: {body[:120]!r}"
        )

    from io import StringIO

    try:
        df = pd.read_csv(StringIO(body))
    except Exception as exc:
        raise SP500DataError(f"stooq CSV parse failed for {ticker}: {exc}") from exc

    if df is None or df.empty or "Date" not in df.columns or "Close" not in df.columns:
        raise SP500DataError(
            f"stooq returned unexpected payload for {ticker}: "
            f"columns={list(df.columns) if df is not None else None}"
        )

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df.index = df.index.tz_localize(None)
    df.index.name = "date"
    return df[["Close"]].astype(np.float32).rename(columns={"Close": "close"})


def _load_with_cache(
    ticker: str,
    source: Literal["yfinance", "stooq"],
    start: str | date | None,
    end: str | date | None,
    cache_dir: Path,
    refresh: bool,
) -> pd.DataFrame:
    path = _cache_path(cache_dir, f"{source}_{ticker}")
    if not refresh:
        cached = _read_cache(path)
        if cached is not None and not cached.empty:
            logger.info("cache hit: %s", path.name)
            return cached

    if source == "yfinance":
        df = _fetch_yfinance(ticker, start, end)
    elif source == "stooq":
        df = _fetch_stooq(ticker, start, end)
    else:
        raise ValueError(f"unsupported source: {source}")

    _write_cache(path, df)
    return df


def _cross_check(
    a: pd.DataFrame,
    b: pd.DataFrame,
    source_a: str,
    source_b: str,
    tolerance_pct: float = TOLERANCE_PCT,
) -> CrossCheckResult:
    """2系列の close の重複期間における乖離率を評価。"""
    if "close" not in a.columns or "close" not in b.columns:
        raise ValueError("both DataFrames must contain 'close' column")

    joined = a[["close"]].rename(columns={"close": "a"}).join(
        b[["close"]].rename(columns={"close": "b"}),
        how="inner",
    ).dropna()

    if joined.empty:
        return CrossCheckResult(
            source_a=source_a,
            source_b=source_b,
            max_diff_pct=float("nan"),
            mean_diff_pct=float("nan"),
            overlap_start=None,
            overlap_end=None,
            passed=False,
            tolerance_pct=tolerance_pct,
        )

    diff_pct = (joined["a"] - joined["b"]).abs() / joined["b"] * 100.0
    max_diff = float(diff_pct.max())
    mean_diff = float(diff_pct.mean())
    passed = max_diff <= tolerance_pct

    if not passed:
        logger.warning(
            "cross-check FAILED: %s vs %s max=%.3f%% mean=%.3f%% tol=%.3f%%",
            source_a,
            source_b,
            max_diff,
            mean_diff,
            tolerance_pct,
        )
    else:
        logger.info(
            "cross-check ok: %s vs %s max=%.3f%% mean=%.3f%%",
            source_a,
            source_b,
            max_diff,
            mean_diff,
        )

    return CrossCheckResult(
        source_a=source_a,
        source_b=source_b,
        max_diff_pct=max_diff,
        mean_diff_pct=mean_diff,
        overlap_start=joined.index.min(),
        overlap_end=joined.index.max(),
        passed=passed,
        tolerance_pct=tolerance_pct,
    )


def fetch_sp500(
    series: Series = "total_return",
    start: str | date | None = "1990-01-01",
    end: str | date | None = None,
    cache_dir: Path | str | None = None,
    refresh: bool = False,
    tolerance_pct: float = TOLERANCE_PCT,
) -> tuple[pd.DataFrame, CrossCheckResult]:
    """S&P 500 の日次データを取得し、二重化検証を実施する。

    Args:
        series: "total_return"（`^SP500TR`, 配当再投資込み）
            または "price"（`^GSPC`, 価格指数）
        start: 取得開始日
        end: 取得終了日（None なら最新）
        cache_dir: ローカルキャッシュ先
        refresh: True の場合キャッシュを無視して再取得
        tolerance_pct: クロスチェック許容乖離率 (%)

    Returns:
        (df, cross_check):
            df は columns=['close'] の DataFrame（index=date, dtype=float32）
            cross_check は CrossCheckResult（price 系列の一次 vs 照合）

    Notes:
        - total_return の場合は `^GSPC`(yfinance) vs `^SPX`(stooq) の
          価格クロスチェックを副次的に実施する（TR 自体の直接的な
          ダブルソースは現実的に難しいため）
    """
    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    _ensure_cache_dir(cache_dir)

    price_yf = _load_with_cache(
        TICKER_PRICE_YF, "yfinance", start, end, cache_dir, refresh
    )
    try:
        price_stooq = _load_with_cache(
            TICKER_PRICE_STOOQ, "stooq", start, end, cache_dir, refresh
        )
        cross = _cross_check(
            price_yf,
            price_stooq,
            source_a=f"yfinance:{TICKER_PRICE_YF}",
            source_b=f"stooq:{TICKER_PRICE_STOOQ}",
            tolerance_pct=tolerance_pct,
        )
    except StooqApikeyRequired as exc:
        logger.warning(
            "stooq apikey required; cross-check degraded. set STOOQ_APIKEY to enable. (%s)",
            exc,
        )
        cross = CrossCheckResult(
            source_a=f"yfinance:{TICKER_PRICE_YF}",
            source_b=f"stooq:{TICKER_PRICE_STOOQ} (unavailable: apikey required)",
            max_diff_pct=float("nan"),
            mean_diff_pct=float("nan"),
            overlap_start=None,
            overlap_end=None,
            passed=False,
            tolerance_pct=tolerance_pct,
        )

    if series == "total_return":
        df = _load_with_cache(
            TICKER_TOTAL_RETURN, "yfinance", start, end, cache_dir, refresh
        )
    elif series == "price":
        df = price_yf
    else:
        raise ValueError(f"unsupported series: {series}")

    return df, cross


def fetch_sp500_monthly(
    series: Series = "total_return",
    start: str | date | None = "1990-01-01",
    end: str | date | None = None,
    cache_dir: Path | str | None = None,
    refresh: bool = False,
    tolerance_pct: float = TOLERANCE_PCT,
) -> tuple[pd.DataFrame, CrossCheckResult]:
    """月次（月末終値）にリサンプルした S&P 500 を返す。

    積立系仮説は月次で十分かつ計算量削減の効果が大きい。
    """
    df, cross = fetch_sp500(
        series=series,
        start=start,
        end=end,
        cache_dir=cache_dir,
        refresh=refresh,
        tolerance_pct=tolerance_pct,
    )
    monthly = df["close"].resample("ME").last().to_frame()
    monthly.index.name = "month_end"
    return monthly, cross
