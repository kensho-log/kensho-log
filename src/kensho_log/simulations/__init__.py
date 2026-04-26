"""検証シミュレーション本体。

各仮説に対応する純粋関数（副作用なし、matplotlib 非依存）を提供する。
プロット／動画生成は別レイヤ（output_layer 等）で行う。
"""

from .age_vs_amount import (
    AgeVsAmountResult,
    simulate_age_vs_amount,
    simulate_age_vs_amount_batch,
)
from .dca_loss_horizon import (
    LossHorizonResult,
    simulate_dca_loss_horizon,
    simulate_dca_loss_horizon_batch,
)
from .timing_race import (
    TimingRaceSummary,
    simulate_timing_race,
    summarize_timing_race,
)

__all__ = [
    "AgeVsAmountResult",
    "LossHorizonResult",
    "TimingRaceSummary",
    "simulate_age_vs_amount",
    "simulate_age_vs_amount_batch",
    "simulate_dca_loss_horizon",
    "simulate_dca_loss_horizon_batch",
    "simulate_timing_race",
    "summarize_timing_race",
]
