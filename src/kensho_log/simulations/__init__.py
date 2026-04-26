"""検証シミュレーション本体。

各仮説に対応する純粋関数（副作用なし、matplotlib 非依存）を提供する。
プロット／動画生成は別レイヤ（output_layer 等）で行う。
"""

from .timing_race import (
    TimingRaceSummary,
    simulate_timing_race,
    summarize_timing_race,
)

__all__ = [
    "TimingRaceSummary",
    "simulate_timing_race",
    "summarize_timing_race",
]
