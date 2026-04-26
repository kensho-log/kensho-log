"""仮説 001 Timing Race の実データ疎通ラン。

- yfinance ^SP500TR / ^GSPC + Stooq ^SPX を取得
- 月末 close にリサンプル
- 3戦略（A/B/C）を simulate_timing_race で計算
- 数値サマリ、データソース差異、コミットハッシュ、環境情報を
  stdout と docs/runs/001-timing-race-YYYYMMDD.md に出力
- プレビュー図を output/figures/001_timing_race_preview.png に保存

メモリ 8GB 制約下で動作する軽量処理のみ。動画生成は別スクリプトで行う。
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kensho_log.data_sources import fetch_sp500_monthly  # noqa: E402
from kensho_log.simulations import (  # noqa: E402
    simulate_timing_race,
    summarize_timing_race,
)

ANNUAL_BUDGET = 1_000_000.0
START_DATE = "1990-01-01"
END_DATE = "2024-12-31"

DOCS_RUNS = REPO_ROOT / "docs" / "runs"
OUTPUT_FIGURES = REPO_ROOT / "output" / "figures"
OUTPUT_TMP = REPO_ROOT / "output" / "tmp"


def _get_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _get_git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except Exception:
        return False


def _format_yen(v: float) -> str:
    return f"{v:>16,.0f}"


def _decide_video(diff_pct_abs: float) -> str:
    if diff_pct_abs >= 10.0:
        return "long-form"
    if diff_pct_abs >= 5.0:
        return "shorts"
    return "repo-log-only"


def main() -> int:
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
    OUTPUT_TMP.mkdir(parents=True, exist_ok=True)
    DOCS_RUNS.mkdir(parents=True, exist_ok=True)

    run_started_at = datetime.now(timezone.utc)
    commit = _get_commit_hash()
    dirty = _get_git_dirty()

    print(f"[run_001] commit={commit} dirty={dirty}")
    print(f"[run_001] fetching S&P500 TR monthly {START_DATE} -> {END_DATE}")

    monthly, cross = fetch_sp500_monthly(
        series="total_return",
        start=START_DATE,
        end=END_DATE,
    )

    print(
        f"[run_001] crosscheck: {cross.source_a} vs {cross.source_b} "
        f"max={cross.max_diff_pct:.4f}% mean={cross.mean_diff_pct:.4f}% "
        f"passed={cross.passed} tol={cross.tolerance_pct}%"
    )
    print(
        f"[run_001] loaded rows={len(monthly)} "
        f"range={monthly.index.min().date()}..{monthly.index.max().date()}"
    )

    close_series = monthly["close"].astype(float)
    result = simulate_timing_race(close_series, annual_budget=ANNUAL_BUDGET)
    summary = summarize_timing_race(result)

    print("[run_001] summary")
    print(f"  years covered      : {summary.years}")
    print(f"  total_invested A   : {_format_yen(summary.total_invested_a)}")
    print(f"  total_invested B   : {_format_yen(summary.total_invested_b)}")
    print(f"  total_invested C   : {_format_yen(summary.total_invested_c)}")
    print(f"  final_value    A   : {_format_yen(summary.final_value_a)}")
    print(f"  final_value    B   : {_format_yen(summary.final_value_b)}")
    print(f"  final_value    C   : {_format_yen(summary.final_value_c)}")
    print(f"  diff (A - C) / C   : {summary.diff_pct_a_vs_c:+.2f}%")
    print(f"  diff (B - C) / C   : {summary.diff_pct_b_vs_c:+.2f}%")

    decision = _decide_video(abs(summary.diff_pct_a_vs_c))
    print(f"  video decision (A vs C): {decision}")

    preview_path = OUTPUT_FIGURES / "001_timing_race_preview.png"
    _save_preview(result, preview_path, commit)
    print(f"[run_001] preview saved: {preview_path.relative_to(REPO_ROOT)}")

    tmp_csv = OUTPUT_TMP / "001_timing_race_monthly.csv"
    result.to_csv(tmp_csv)
    print(f"[run_001] monthly result: {tmp_csv.relative_to(REPO_ROOT)}")

    log_path = _write_run_log(
        summary=summary,
        cross=cross,
        commit=commit,
        dirty=dirty,
        run_started_at=run_started_at,
        rows=len(monthly),
        decision=decision,
    )
    print(f"[run_001] run log     : {log_path.relative_to(REPO_ROOT)}")
    return 0


def _save_preview(result, path: Path, commit: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)
    ax.plot(result.index, result["A_value"], label="A: perfect timing", color="#8b0000", linewidth=1.2)
    ax.plot(result.index, result["B_value"], label="B: monthly DCA",   color="#9e9e9e", linewidth=1.2)
    ax.plot(result.index, result["C_value"], label="C: worst timing",  color="#1a3a6b", linewidth=1.2)
    ax.plot(result.index, result["A_invested_cum"], label="cumulative invested (same for A/B/C)",
            color="#3b3b3b", linewidth=0.8, linestyle="--")
    ax.set_title("Hypothesis 001 - Timing Race (S&P 500 TR, 1990-2024)")
    ax.set_xlabel("")
    ax.set_ylabel("portfolio value (JPY, same-currency abstraction)")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.text(
        0.99, 0.01,
        f"commit={commit[:12]}",
        ha="right", va="bottom", fontsize=7, family="monospace", color="#555555",
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_run_log(
    *,
    summary,
    cross,
    commit: str,
    dirty: bool,
    run_started_at: datetime,
    rows: int,
    decision: str,
) -> Path:
    date_tag = run_started_at.strftime("%Y%m%d")
    path = DOCS_RUNS / f"001-timing-race-{date_tag}.md"

    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "runner": "local",
    }

    def _nan_to_none(x: float) -> float | None:
        import math
        return None if (isinstance(x, float) and math.isnan(x)) else x

    metrics = {
        "hypothesis_id": "001",
        "commit": commit,
        "git_dirty": dirty,
        "run_started_at_utc": run_started_at.isoformat(timespec="seconds"),
        "period_start": str(summary.period_start.date()),
        "period_end": str(summary.period_end.date()),
        "years": summary.years,
        "rows": rows,
        "annual_budget_jpy": ANNUAL_BUDGET,
        "final_value_a": summary.final_value_a,
        "final_value_b": summary.final_value_b,
        "final_value_c": summary.final_value_c,
        "total_invested_a": summary.total_invested_a,
        "total_invested_b": summary.total_invested_b,
        "total_invested_c": summary.total_invested_c,
        "diff_pct_a_vs_c": summary.diff_pct_a_vs_c,
        "diff_pct_b_vs_c": summary.diff_pct_b_vs_c,
        "video_decision": decision,
        "crosscheck": {
            "source_a": cross.source_a,
            "source_b": cross.source_b,
            "max_diff_pct": _nan_to_none(cross.max_diff_pct),
            "mean_diff_pct": _nan_to_none(cross.mean_diff_pct),
            "tolerance_pct": cross.tolerance_pct,
            "passed": cross.passed,
            "overlap_start": None if cross.overlap_start is None else str(cross.overlap_start.date()),
            "overlap_end": None if cross.overlap_end is None else str(cross.overlap_end.date()),
        },
        "env": env,
    }

    lines: list[str] = []
    lines.append(f"# Run 001 Timing Race - {date_tag}")
    lines.append("")
    lines.append(f"- commit: `{commit}` (dirty={dirty})")
    lines.append(f"- run_started_at: `{run_started_at.isoformat(timespec='seconds')}`")
    lines.append(f"- period: {summary.period_start.date()} to {summary.period_end.date()} ({summary.years} years, {rows} monthly rows)")
    lines.append(f"- annual_budget: {int(ANNUAL_BUDGET):,} JPY")
    lines.append(f"- python: {env['python']} / {env['platform']}")
    lines.append("")
    lines.append("## Crosscheck (yfinance vs stooq, price series)")
    lines.append("")
    lines.append(f"- sources: {cross.source_a} vs {cross.source_b}")
    lines.append(f"- overlap: {cross.overlap_start} to {cross.overlap_end}")
    lines.append(f"- max diff: {cross.max_diff_pct:.4f}% / mean diff: {cross.mean_diff_pct:.4f}%")
    lines.append(f"- tolerance: {cross.tolerance_pct}% / passed: {cross.passed}")
    lines.append("")
    lines.append("## Final Values")
    lines.append("")
    lines.append("| strategy | total invested | final value | vs C |")
    lines.append("| --- | ---: | ---: | ---: |")
    lines.append(
        f"| A perfect timing | {summary.total_invested_a:,.0f} | "
        f"{summary.final_value_a:,.0f} | {summary.diff_pct_a_vs_c:+.2f}% |"
    )
    lines.append(
        f"| B monthly DCA | {summary.total_invested_b:,.0f} | "
        f"{summary.final_value_b:,.0f} | {summary.diff_pct_b_vs_c:+.2f}% |"
    )
    lines.append(
        f"| C worst timing | {summary.total_invested_c:,.0f} | "
        f"{summary.final_value_c:,.0f} | 0.00% |"
    )
    lines.append("")
    lines.append("## Video Decision")
    lines.append("")
    lines.append(
        f"- |diff(A vs C)| = {abs(summary.diff_pct_a_vs_c):.2f}% -> **{decision}**"
    )
    lines.append("")
    lines.append("## Machine-readable metrics (JSON)")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False))
    lines.append("```")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("- ドル建てのまま JPY として便宜的に扱っている。為替影響は未考慮。")
    lines.append("- 税制（配当20.315%、売却益）・取引コスト・信託報酬は未考慮。")
    lines.append("- `^SP500TR` の利用可能範囲（1988年以降）に依存。")
    lines.append("- 前月見るのではなく「年内 min/max」のハインドサイトを前提とする理想化モデル。")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
