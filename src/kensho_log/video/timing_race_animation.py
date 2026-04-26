"""仮説 001 Timing Race の動画フレーム生成 + FFmpeg 結合。

設計方針:
- 純関数 write_timing_race_frames(result, frames_dir, ...): フレーム PNG を書き出し
- 外部依存 render_timing_race_video(...): 上記 + FFmpeg invocation
- MoviePy 等の重量ライブラリは使用禁止（FFmpeg は subprocess 直叩き）
- メモリ常駐最小化のため、1 フレームごとに Agg backend で figure を close する
- Y/X 軸固定（チャンネル仕様: 軸は不動、ズーム不可）

描画要素（PoC minimal）:
- 背景: 極めて暗いグレー (#0e0f12)
- 3 本線: A=えび茶、B=アッシュグレー、C=深紺
- 累積投資の点線（薄グレー）
- 上部タイトル（明朝白）: Hypothesis 001 - Timing Race
- 右上: 日付
- 右下: 3 戦略の評価額（モノスペース、整数のみ）
- 下部中央: 現在月の要約テキスト（明朝白）
- 右下極小: コミットハッシュ（再現性担保）
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import rcParams  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402

import pandas as pd  # noqa: E402

logger = logging.getLogger(__name__)

FRAMES_PER_SECOND_DEFAULT = 30
FIGSIZE_720P = (12.80, 7.20)
DPI_720P = 100

BG_COLOR = "#0e0f12"
FG_COLOR = "#eaeaea"
GRID_COLOR = "#2a2a2a"
COLOR_A = "#b46a6a"
COLOR_B = "#9e9e9e"
COLOR_C = "#6b88b4"
COLOR_INVESTED = "#5a5a5a"

MINCHO_CANDIDATES = ("Yu Mincho", "MS Mincho", "BIZ UDMincho", "Noto Serif CJK JP")
MONO_CANDIDATES = ("Consolas", "Courier New", "DejaVu Sans Mono")


class FfmpegNotFoundError(RuntimeError):
    """ffmpeg 実行ファイルが PATH 上に見つからない場合に送出。"""


@dataclass(frozen=True)
class _FontPair:
    mincho: FontProperties
    mono: FontProperties


def _pick_font(candidates: tuple[str, ...], fallback_family: str) -> FontProperties:
    from matplotlib.font_manager import fontManager

    names = {f.name for f in fontManager.ttflist}
    for c in candidates:
        if c in names:
            return FontProperties(family=c)
    logger.warning(
        "none of %s found; falling back to generic %s", candidates, fallback_family
    )
    return FontProperties(family=fallback_family)


def _get_fonts() -> _FontPair:
    return _FontPair(
        mincho=_pick_font(MINCHO_CANDIDATES, "serif"),
        mono=_pick_font(MONO_CANDIDATES, "monospace"),
    )


def _apply_dark_style() -> None:
    rcParams["axes.facecolor"] = BG_COLOR
    rcParams["figure.facecolor"] = BG_COLOR
    rcParams["savefig.facecolor"] = BG_COLOR
    rcParams["axes.edgecolor"] = GRID_COLOR
    rcParams["axes.labelcolor"] = FG_COLOR
    rcParams["xtick.color"] = FG_COLOR
    rcParams["ytick.color"] = FG_COLOR
    rcParams["grid.color"] = GRID_COLOR
    rcParams["grid.alpha"] = 0.4
    rcParams["text.color"] = FG_COLOR


def race_frame_indices(
    n_rows: int,
    stride: int = 1,
    max_frames: int | None = None,
    *,
    num_frames: int | None = None,
) -> list[int]:
    """`write_timing_race_frames` と同じ行インデックス列。

    ``num_frames`` を指定した場合は ``stride`` / ``max_frames`` を無視し、
    ``[0, n_rows-1]`` から num_frames 個の行をほぼ等間隔に選ぶ（最後の行は必ず含む）。
    フレーム数 > n_rows の場合は同じ行を複数回サンプリングする（時間軸を伸ばす効果）。
    """
    if n_rows < 1:
        raise ValueError("n_rows must be >= 1")
    if num_frames is not None:
        if num_frames < 1:
            raise ValueError("num_frames must be >= 1")
        if num_frames == 1:
            return [n_rows - 1]
        idxs: list[int] = []
        last = n_rows - 1
        for k in range(num_frames):
            r = k / (num_frames - 1)
            i = int(round(r * last))
            if i < 0:
                i = 0
            elif i > last:
                i = last
            idxs.append(i)
        idxs[-1] = last
        return idxs
    if stride < 1:
        raise ValueError("stride must be >= 1")
    idxs = list(range(0, n_rows, stride))
    if idxs[-1] != n_rows - 1:
        idxs.append(n_rows - 1)
    if max_frames is not None and max_frames > 0:
        idxs = idxs[:max_frames]
    return idxs


def _validate_result(result: pd.DataFrame) -> pd.DataFrame:
    required = {
        "close", "A_value", "B_value", "C_value",
        "A_invested_cum", "B_invested_cum", "C_invested_cum",
    }
    missing = required - set(result.columns)
    if missing:
        raise ValueError(f"result is missing columns: {sorted(missing)}")
    if result.empty:
        raise ValueError("result must not be empty")
    if not isinstance(result.index, pd.DatetimeIndex):
        raise TypeError("result.index must be a DatetimeIndex")
    return result.sort_index()


def _format_jpy(v: float) -> str:
    return f"{int(round(v)):,}"


def _draw_footer_chrome(fig, commit_hash: str, fonts: _FontPair) -> None:
    fig.text(
        0.99, 0.015, f"commit {commit_hash[:12]}",
        ha="right", va="bottom", fontsize=8, color="#777777",
        fontproperties=fonts.mono,
    )
    fig.text(
        0.01, 0.015, "kensho-log  verification log",
        ha="left", va="bottom", fontsize=8, color="#777777",
        fontproperties=fonts.mincho,
    )


def _wrap_telop_lines(text: str, max_chars: int) -> list[str]:
    t = (text or "").strip()
    if not t:
        return []
    lines: list[str] = []
    rest = t
    while rest:
        if len(rest) <= max_chars:
            lines.append(rest)
            break
        lines.append(rest[:max_chars])
        rest = rest[max_chars:].lstrip()
    return lines


def _draw_bottom_telop(
    fig, text: str, fonts: _FontPair, *, alpha: float = 1.0, fontsize: int = 11,
) -> None:
    if not (text and text.strip()):
        return
    lines = _wrap_telop_lines(text.strip(), 32)[:3]
    y0 = 0.10
    for j, line in enumerate(lines):
        fig.text(
            0.5, y0 - j * 0.032, line,
            ha="center", va="top", fontsize=fontsize, color=FG_COLOR,
            fontproperties=fonts.mincho, alpha=alpha, clip_on=False,
        )


def _add_right_character_placeholder(
    fig, fonts: _FontPair, label: str, *, alpha: float = 1.0,
) -> None:
    axr = fig.add_axes([0.70, 0.20, 0.28, 0.70])
    axr.set_facecolor("#1a1a1c")
    axr.patch.set_edgecolor(GRID_COLOR)
    axr.patch.set_linewidth(0.5)
    axr.set_xticks([])
    axr.set_yticks([])
    axr.text(
        0.5, 0.5, label,
        ha="center", va="center", fontsize=10, color="#5a5a5a", alpha=alpha,
        fontproperties=fonts.mincho, transform=axr.transAxes,
    )


def write_timing_race_frames(
    result: pd.DataFrame,
    frames_dir: Path | str,
    commit_hash: str,
    stride: int = 1,
    max_frames: int | None = None,
    dpi: int = DPI_720P,
    figsize: tuple[float, float] = FIGSIZE_720P,
    title: str = "Hypothesis 001 - Timing Race",
    subtitle: str = "S&P 500 Total Return",
    filename_template: str = "frame_{:05d}.png",
    start_index: int = 0,
    event_markers: list[tuple[pd.Timestamp, str]] | None = None,
    three_pane: bool = False,
    telop_by_frame: list[str] | None = None,
    placeholder_label: str = "log  (CG placeholder)",
    num_frames: int | None = None,
) -> list[Path]:
    """result DataFrame から PNG フレーム群を frames_dir に生成する。

    Args:
        result: simulate_timing_race の出力（DatetimeIndex、必要列を含む）
        frames_dir: フレーム出力先
        commit_hash: 映像字幕に焼き込むコミットハッシュ
        stride: 何行ごとに 1 フレームを作るか（デフォルト 1 = 月次そのまま）
        max_frames: 生成フレーム上限（デバッグ / スモークテスト用）
        dpi / figsize: 出力解像度（figsize * dpi = pixel size）
        title / subtitle: 静的タイトル
        filename_template: 連番フォーマット
        start_index: 出力ファイル名の開始番号（story モードで複数フェーズ連結時に使用）
        event_markers: (Timestamp, ラベル) のリスト。該当月以降、縦破線でマーク。

    Returns:
        生成されたフレームの Path リスト（昇順）
    """
    if num_frames is None and stride < 1:
        raise ValueError("stride must be >= 1")

    result = _validate_result(result)
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    _apply_dark_style()
    fonts = _get_fonts()

    x_full = result.index
    y_max = float(
        max(
            result["A_value"].max(),
            result["B_value"].max(),
            result["C_value"].max(),
            result["A_invested_cum"].max(),
        )
    )
    y_cap = y_max * 1.08 if y_max > 0 else 1.0

    idxs = race_frame_indices(
        len(result), stride, max_frames, num_frames=num_frames
    )

    n_race = len(idxs)
    if three_pane and telop_by_frame is not None and len(telop_by_frame) != n_race:
        raise ValueError(
            f"telop_by_frame must have {n_race} lines (one per frame), got {len(telop_by_frame)}"
        )
    tlines = (telop_by_frame if telop_by_frame is not None else [""] * n_race) if three_pane else []

    marker_list = list(event_markers or [])

    produced: list[Path] = []
    for fi, row_i in enumerate(idxs):
        slice_end = row_i + 1
        visible = result.iloc[:slice_end]

        if not three_pane:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            fig = plt.figure(figsize=figsize, dpi=dpi)
            fig.patch.set_facecolor(BG_COLOR)
            ax = fig.add_axes((0.08, 0.25, 0.60, 0.64))
        ax.set_facecolor(BG_COLOR)

        ax.plot(x_full, [float("nan")] * len(x_full))
        ax.plot(
            visible.index, visible["A_invested_cum"],
            color=COLOR_INVESTED, linewidth=0.9, linestyle="--",
            label="cumulative invested",
        )
        ax.plot(
            visible.index, visible["C_value"],
            color=COLOR_C, linewidth=1.4,
            label="C  worst timing",
        )
        ax.plot(
            visible.index, visible["B_value"],
            color=COLOR_B, linewidth=1.4,
            label="B  monthly DCA",
        )
        ax.plot(
            visible.index, visible["A_value"],
            color=COLOR_A, linewidth=1.4,
            label="A  perfect timing",
        )

        ax.set_xlim(x_full.min(), x_full.max())
        ax.set_ylim(0, y_cap)
        ax.grid(True, linewidth=0.3, alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(GRID_COLOR)
        ax.spines["bottom"].set_color(GRID_COLOR)

        current_date = visible.index[-1]

        for m_date, m_label in marker_list:
            if current_date >= m_date and m_date >= x_full.min() and m_date <= x_full.max():
                ax.axvline(
                    x=m_date, color="#d27171", linewidth=0.8,
                    linestyle=(0, (2, 3)), alpha=0.55,
                )
                ax.text(
                    m_date, y_cap * 0.04, f" {m_label}",
                    ha="left", va="bottom", fontsize=8.5, color="#c59090",
                    fontproperties=fonts.mincho, alpha=0.85,
                )

        if not three_pane:
            fig.text(
                0.03, 0.955, title,
                ha="left", va="top", fontsize=18, color=FG_COLOR,
                fontproperties=fonts.mincho,
            )
            fig.text(
                0.03, 0.915, subtitle,
                ha="left", va="top", fontsize=11, color="#aaaaaa",
                fontproperties=fonts.mincho,
            )

            fig.text(
                0.97, 0.955, current_date.strftime("%Y-%m"),
                ha="right", va="top", fontsize=13, color=FG_COLOR,
                fontproperties=fonts.mono,
            )

        last = visible.iloc[-1]
        if not three_pane:
            box_x = 0.98
            box_y = 0.42
            line_h = 0.06

            for j, (label, value, color) in enumerate(
                [
                    ("A", last["A_value"], COLOR_A),
                    ("B", last["B_value"], COLOR_B),
                    ("C", last["C_value"], COLOR_C),
                ]
            ):
                fig.text(
                    box_x - 0.16, box_y - j * line_h, label,
                    ha="right", va="center", fontsize=12, color=color,
                    fontproperties=fonts.mincho,
                )
                fig.text(
                    box_x, box_y - j * line_h, _format_jpy(value),
                    ha="right", va="center", fontsize=15, color=FG_COLOR,
                    fontproperties=fonts.mono,
                )

            fig.text(
                box_x, box_y - 3 * line_h, "invested",
                ha="right", va="center", fontsize=9, color="#888888",
                fontproperties=fonts.mincho,
            )
            fig.text(
                box_x, box_y - 3.7 * line_h, _format_jpy(last["A_invested_cum"]),
                ha="right", va="center", fontsize=11, color="#aaaaaa",
                fontproperties=fonts.mono,
            )
        else:
            fig.text(
                0.08, 0.955, title,
                ha="left", va="top", fontsize=15, color=FG_COLOR,
                fontproperties=fonts.mincho,
            )
            fig.text(
                0.08, 0.915, subtitle,
                ha="left", va="top", fontsize=9, color="#aaaaaa",
                fontproperties=fonts.mincho,
            )
            fig.text(
                0.64, 0.955, current_date.strftime("%Y-%m"),
                ha="right", va="top", fontsize=11, color=FG_COLOR,
                fontproperties=fonts.mono,
            )
            y_rows = (0.58, 0.45, 0.32)
            for j, (lab, value, col) in enumerate(
                [
                    ("A", last["A_value"], COLOR_A),
                    ("B", last["B_value"], COLOR_B),
                    ("C", last["C_value"], COLOR_C),
                ]
            ):
                y = y_rows[j]
                ax.text(
                    0.95, y + 0.03, lab,
                    ha="right", va="bottom", fontsize=9, color=col,
                    fontproperties=fonts.mincho, transform=ax.transAxes,
                )
                ax.text(
                    0.95, y, _format_jpy(value),
                    ha="right", va="top", fontsize=10, color=FG_COLOR,
                    fontproperties=fonts.mono, transform=ax.transAxes,
                )
            ax.text(
                0.95, 0.16, "invested", ha="right", va="top",
                fontsize=7, color="#888888", fontproperties=fonts.mincho,
                transform=ax.transAxes,
            )
            ax.text(
                0.95, 0.1, _format_jpy(last["A_invested_cum"]),
                ha="right", va="top", fontsize=8, color="#aaaaaa",
                fontproperties=fonts.mono, transform=ax.transAxes,
            )
            _add_right_character_placeholder(fig, fonts, placeholder_label)
            if tlines[fi].strip():
                _draw_bottom_telop(fig, tlines[fi], fonts, fontsize=9)

        ax.get_legend().remove() if ax.get_legend() else None

        _draw_footer_chrome(fig, commit_hash, fonts)

        frame_path = frames_dir / filename_template.format(start_index + fi)
        fig.savefig(frame_path, dpi=dpi)
        plt.close(fig)
        produced.append(frame_path)

    logger.info("wrote %d frames to %s", len(produced), frames_dir)
    return produced


def _fade_alpha(i: int, total: int, fade_in: int) -> float:
    """0..fade_in で 0→1、以降 1 のキッカリしたフェードインカーブ。"""
    if fade_in <= 0:
        return 1.0
    if i >= fade_in:
        return 1.0
    return max(0.0, min(1.0, i / fade_in))


def _fresh_canvas(figsize: tuple[float, float], dpi: int):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(BG_COLOR)
    return fig


def write_titlecard_frames(
    lines: list[str],
    frames_dir: Path | str,
    commit_hash: str,
    num_frames: int,
    *,
    start_index: int = 0,
    fade_in_frames: int = 15,
    dpi: int = DPI_720P,
    figsize: tuple[float, float] = FIGSIZE_720P,
    filename_template: str = "frame_{:05d}.png",
    font_size: int = 30,
    line_spacing: float = 0.09,
    three_pane: bool = False,
    bottom_telop: str = "",
    placeholder_label: str = "log  (CG placeholder)",
) -> list[Path]:
    """黒背景のタイトルカード（疑問提示用）を num_frames 枚生成する。

    Args:
        lines: 中央に縦に並べる明朝テキスト行
        num_frames: 生成フレーム数
        fade_in_frames: 先頭で 0→1 にアルファを上げるフレーム数
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if fade_in_frames < 0:
        raise ValueError("fade_in_frames must be >= 0")

    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    _apply_dark_style()
    fonts = _get_fonts()

    produced: list[Path] = []
    n = len(lines)
    center_y = 0.5 + (n - 1) * line_spacing / 2
    for i in range(num_frames):
        alpha = _fade_alpha(i, num_frames, fade_in_frames)
        fig = _fresh_canvas(figsize, dpi)
        if not three_pane:
            for j, text in enumerate(lines):
                y = center_y - j * line_spacing
                fig.text(
                    0.5, y, text,
                    ha="center", va="center",
                    fontsize=font_size, color=FG_COLOR,
                    fontproperties=fonts.mincho, alpha=alpha,
                )
        else:
            ax_left = fig.add_axes((0.06, 0.20, 0.62, 0.70))
            ax_left.set_facecolor(BG_COLOR)
            ax_left.set_xlim(0, 1)
            ax_left.set_ylim(0, 1)
            ax_left.axis("off")
            cy = 0.5 + (n - 1) * (line_spacing * 0.6) / 2
            fs = int(font_size * 0.85) if n > 2 else font_size
            for j, text in enumerate(lines):
                yl = cy - j * line_spacing * 0.6
                ax_left.text(
                    0.5, yl, text,
                    ha="center", va="center", transform=ax_left.transAxes,
                    fontsize=fs, color=FG_COLOR, fontproperties=fonts.mincho, alpha=alpha,
                )
            _add_right_character_placeholder(fig, fonts, placeholder_label, alpha=alpha)
            if bottom_telop:
                _draw_bottom_telop(fig, bottom_telop, fonts, alpha=alpha, fontsize=10)
        _draw_footer_chrome(fig, commit_hash, fonts)
        frame_path = frames_dir / filename_template.format(start_index + i)
        fig.savefig(frame_path, dpi=dpi)
        plt.close(fig)
        produced.append(frame_path)
    return produced


def write_conditions_frames(
    conditions: list[tuple[str, str, str]],
    frames_dir: Path | str,
    commit_hash: str,
    num_frames: int,
    *,
    heading: str = "検証条件",
    start_index: int = 0,
    fade_in_frames: int = 10,
    dpi: int = DPI_720P,
    figsize: tuple[float, float] = FIGSIZE_720P,
    filename_template: str = "frame_{:05d}.png",
    three_pane: bool = False,
    bottom_telop: str = "",
    placeholder_label: str = "log  (CG placeholder)",
) -> list[Path]:
    """条件カード。conditions = [(symbol, label, description), ...]。"""
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if not conditions:
        raise ValueError("conditions must not be empty")

    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    _apply_dark_style()
    fonts = _get_fonts()

    palette = [COLOR_A, COLOR_B, COLOR_C, "#8b8b8b", "#8b8b8b"]

    produced: list[Path] = []
    for i in range(num_frames):
        alpha = _fade_alpha(i, num_frames, fade_in_frames)
        fig = _fresh_canvas(figsize, dpi)
        if not three_pane:
            fig.text(
                0.5, 0.82, heading,
                ha="center", va="center", fontsize=22, color=FG_COLOR,
                fontproperties=fonts.mincho, alpha=alpha,
            )

            n = len(conditions)
            row_h = 0.11
            top_y = 0.62
            for k, (symbol, label, desc) in enumerate(conditions):
                y = top_y - k * row_h
                color = palette[k] if k < len(palette) else FG_COLOR
                fig.text(
                    0.18, y, symbol,
                    ha="left", va="center", fontsize=34, color=color,
                    fontproperties=fonts.mincho, alpha=alpha,
                )
                fig.text(
                    0.27, y + 0.015, label,
                    ha="left", va="center", fontsize=18, color=FG_COLOR,
                    fontproperties=fonts.mincho, alpha=alpha,
                )
                fig.text(
                    0.27, y - 0.025, desc,
                    ha="left", va="center", fontsize=13, color="#9aa0a6",
                    fontproperties=fonts.mincho, alpha=alpha,
                )
        else:
            axc = fig.add_axes((0.04, 0.18, 0.64, 0.76))
            axc.set_facecolor(BG_COLOR)
            axc.set_xlim(0, 1)
            axc.set_ylim(0, 1)
            axc.axis("off")
            axc.text(
                0.5, 0.90, heading,
                ha="center", va="center", fontsize=19, color=FG_COLOR,
                fontproperties=fonts.mincho, alpha=alpha, transform=axc.transAxes,
            )
            row_h, top_y = 0.20, 0.68
            n = len(conditions)
            for k, (symbol, label, desc) in enumerate(conditions):
                y = top_y - k * row_h
                color = palette[k] if k < len(palette) else FG_COLOR
                axc.text(
                    0.08, y, symbol,
                    ha="left", va="center", fontsize=30, color=color,
                    fontproperties=fonts.mincho, alpha=alpha, transform=axc.transAxes,
                )
                axc.text(
                    0.22, y + 0.04, label,
                    ha="left", va="center", fontsize=15, color=FG_COLOR,
                    fontproperties=fonts.mincho, alpha=alpha, transform=axc.transAxes,
                )
                axc.text(
                    0.22, y, desc,
                    ha="left", va="top", fontsize=10, color="#9aa0a6",
                    fontproperties=fonts.mincho, alpha=alpha, transform=axc.transAxes,
                )
            _add_right_character_placeholder(fig, fonts, placeholder_label, alpha=alpha)
            if bottom_telop:
                _draw_bottom_telop(fig, bottom_telop, fonts, alpha=alpha, fontsize=10)
        _draw_footer_chrome(fig, commit_hash, fonts)
        frame_path = frames_dir / filename_template.format(start_index + i)
        fig.savefig(frame_path, dpi=dpi)
        plt.close(fig)
        produced.append(frame_path)
    return produced


def write_finale_frames(
    final_values: dict[str, float],
    invested_total: float,
    frames_dir: Path | str,
    commit_hash: str,
    num_frames: int,
    *,
    odometer_ratio: float = 0.55,
    heading: str = "最終評価額",
    start_index: int = 0,
    fade_in_frames: int = 10,
    dpi: int = DPI_720P,
    figsize: tuple[float, float] = FIGSIZE_720P,
    filename_template: str = "frame_{:05d}.png",
    three_pane: bool = False,
    bottom_telop: str = "",
    placeholder_label: str = "log  (CG placeholder)",
) -> list[Path]:
    """フィナーレカード。
    final_values: {"A": ..., "B": ..., "C": ...}
    odometer_ratio: 0→final までカウントアップに使うフレーム割合。
    """
    required_keys = {"A", "B", "C"}
    if set(final_values) != required_keys:
        raise ValueError(f"final_values keys must be {required_keys}, got {set(final_values)}")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if not 0.0 < odometer_ratio <= 1.0:
        raise ValueError("odometer_ratio must be in (0, 1]")

    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    _apply_dark_style()
    fonts = _get_fonts()

    fa = float(final_values["A"])
    fb = float(final_values["B"])
    fc = float(final_values["C"])

    odometer_end = max(1, int(num_frames * odometer_ratio))

    colors = [COLOR_A, COLOR_B, COLOR_C]
    symbols = ["A", "B", "C"]
    labels = ["完璧タイミング", "毎月積立", "最悪タイミング"]
    targets = [fa, fb, fc]

    produced: list[Path] = []
    for i in range(num_frames):
        alpha = _fade_alpha(i, num_frames, fade_in_frames)
        if i < odometer_end:
            t = (i + 1) / odometer_end
            t = t * t
            running = [t * v for v in targets]
        else:
            running = list(targets)

        fig = _fresh_canvas(figsize, dpi)
        if not three_pane:
            fig.text(
                0.5, 0.88, heading,
                ha="center", va="center", fontsize=20, color="#b8b8b8",
                fontproperties=fonts.mincho, alpha=alpha,
            )

            col_xs = [0.22, 0.50, 0.78]
            for k in range(3):
                fig.text(
                    col_xs[k], 0.70, symbols[k],
                    ha="center", va="center", fontsize=26, color=colors[k],
                    fontproperties=fonts.mincho, alpha=alpha,
                )
                fig.text(
                    col_xs[k], 0.64, labels[k],
                    ha="center", va="center", fontsize=12, color="#9aa0a6",
                    fontproperties=fonts.mincho, alpha=alpha,
                )
                fig.text(
                    col_xs[k], 0.48, f"¥{int(round(running[k])):,}",
                    ha="center", va="center", fontsize=32, color=FG_COLOR,
                    fontproperties=fonts.mono, alpha=alpha,
                )

            diff_a_b = targets[0] - targets[1]
            diff_running = running[0] - running[1]
            fig.text(
                0.5, 0.28, "差額  A − B",
                ha="center", va="center", fontsize=14, color="#9aa0a6",
                fontproperties=fonts.mincho, alpha=alpha,
            )
            sign = "+" if diff_running >= 0 else "−"
            fig.text(
                0.5, 0.22, f"{sign}¥{int(round(abs(diff_running))):,}",
                ha="center", va="center", fontsize=36, color=FG_COLOR,
                fontproperties=fonts.mono, alpha=alpha,
            )
            fig.text(
                0.5, 0.15, f"投資元本 ¥{int(round(invested_total)):,}",
                ha="center", va="center", fontsize=11, color="#777777",
                fontproperties=fonts.mincho, alpha=alpha,
            )
        else:
            axc = fig.add_axes((0.04, 0.12, 0.64, 0.80))
            axc.set_facecolor(BG_COLOR)
            axc.set_xlim(0, 1)
            axc.set_ylim(0, 1)
            axc.axis("off")
            axc.text(
                0.5, 0.90, heading,
                ha="center", va="center", fontsize=16, color="#b8b8b8",
                fontproperties=fonts.mincho, alpha=alpha, transform=axc.transAxes,
            )
            col_xs = (0.18, 0.50, 0.82)
            for k in range(3):
                axc.text(
                    col_xs[k], 0.70, symbols[k],
                    ha="center", va="center", fontsize=22, color=colors[k],
                    fontproperties=fonts.mincho, alpha=alpha, transform=axc.transAxes,
                )
                axc.text(
                    col_xs[k], 0.64, labels[k],
                    ha="center", va="center", fontsize=9, color="#9aa0a6",
                    fontproperties=fonts.mincho, alpha=alpha, transform=axc.transAxes,
                )
                axc.text(
                    col_xs[k], 0.44, f"¥{int(round(running[k])):,}",
                    ha="center", va="center", fontsize=20, color=FG_COLOR,
                    fontproperties=fonts.mono, alpha=alpha, transform=axc.transAxes,
                )
            diff_running = running[0] - running[1]
            axc.text(
                0.5, 0.25, "差額  A − B",
                ha="center", va="center", fontsize=11, color="#9aa0a6",
                fontproperties=fonts.mincho, alpha=alpha, transform=axc.transAxes,
            )
            sign = "+" if diff_running >= 0 else "−"
            axc.text(
                0.5, 0.16, f"{sign}¥{int(round(abs(diff_running))):,}",
                ha="center", va="center", fontsize=24, color=FG_COLOR,
                fontproperties=fonts.mono, alpha=alpha, transform=axc.transAxes,
            )
            axc.text(
                0.5, 0.06, f"投資元本 ¥{int(round(invested_total)):,}",
                ha="center", va="center", fontsize=9, color="#777777",
                fontproperties=fonts.mincho, alpha=alpha, transform=axc.transAxes,
            )
            _add_right_character_placeholder(fig, fonts, placeholder_label, alpha=alpha)
            if bottom_telop:
                _draw_bottom_telop(fig, bottom_telop, fonts, alpha=alpha, fontsize=9)

        _draw_footer_chrome(fig, commit_hash, fonts)
        frame_path = frames_dir / filename_template.format(start_index + i)
        fig.savefig(frame_path, dpi=dpi)
        plt.close(fig)
        produced.append(frame_path)
    return produced


def write_limits_frames(
    bullets: list[str],
    frames_dir: Path | str,
    commit_hash: str,
    num_frames: int,
    *,
    heading: str = "本検証の限界",
    start_index: int = 0,
    fade_in_frames: int = 10,
    dpi: int = DPI_720P,
    figsize: tuple[float, float] = FIGSIZE_720P,
    filename_template: str = "frame_{:05d}.png",
    three_pane: bool = False,
    bottom_telop: str = "",
    placeholder_label: str = "log  (CG placeholder)",
) -> list[Path]:
    """「本検証の限界」カード。bullets を箇条書きで描画する。"""
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if not bullets:
        raise ValueError("bullets must not be empty")

    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    _apply_dark_style()
    fonts = _get_fonts()

    produced: list[Path] = []
    for i in range(num_frames):
        alpha = _fade_alpha(i, num_frames, fade_in_frames)
        fig = _fresh_canvas(figsize, dpi)
        if not three_pane:
            fig.text(
                0.5, 0.85, heading,
                ha="center", va="center", fontsize=22, color=FG_COLOR,
                fontproperties=fonts.mincho, alpha=alpha,
            )
            row_h = 0.10
            top_y = 0.65
            for k, b in enumerate(bullets):
                y = top_y - k * row_h
                fig.text(
                    0.18, y, "・",
                    ha="left", va="center", fontsize=16, color="#888888",
                    fontproperties=fonts.mincho, alpha=alpha,
                )
                fig.text(
                    0.22, y, b,
                    ha="left", va="center", fontsize=14, color="#cccccc",
                    fontproperties=fonts.mincho, alpha=alpha,
                )
        else:
            axc = fig.add_axes((0.04, 0.12, 0.64, 0.80))
            axc.set_facecolor(BG_COLOR)
            axc.set_xlim(0, 1)
            axc.set_ylim(0, 1)
            axc.axis("off")
            axc.text(
                0.5, 0.88, heading,
                ha="center", va="center", fontsize=19, color=FG_COLOR,
                fontproperties=fonts.mincho, alpha=alpha, transform=axc.transAxes,
            )
            row_h, top_y = 0.11, 0.68
            for k, b in enumerate(bullets):
                y = top_y - k * row_h
                axc.text(
                    0.06, y, "・",
                    ha="left", va="center", fontsize=13, color="#888888",
                    fontproperties=fonts.mincho, alpha=alpha, transform=axc.transAxes,
                )
                axc.text(
                    0.1, y, b,
                    ha="left", va="center", fontsize=9, color="#cccccc",
                    fontproperties=fonts.mincho, alpha=alpha, transform=axc.transAxes,
                )
            _add_right_character_placeholder(fig, fonts, placeholder_label, alpha=alpha)
            if bottom_telop:
                _draw_bottom_telop(fig, bottom_telop, fonts, alpha=alpha, fontsize=9)
        _draw_footer_chrome(fig, commit_hash, fonts)
        frame_path = frames_dir / filename_template.format(start_index + i)
        fig.savefig(frame_path, dpi=dpi)
        plt.close(fig)
        produced.append(frame_path)
    return produced


def _ffmpeg_binary() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        raise FfmpegNotFoundError(
            "ffmpeg not found on PATH. install ffmpeg and ensure it is on PATH."
        )
    return path


def _build_ffmpeg_cmd(
    frames_glob_pattern: str,
    output_path: Path,
    fps: int,
    freeze_last_seconds: float,
) -> list[str]:
    """個別 PNG を入力に H.264 で mp4 を出力する ffmpeg コマンドを組み立てる。

    最終フレームを freeze_last_seconds 秒ホールドする tpad を付加する。
    """
    ffmpeg = _ffmpeg_binary()
    vf = (
        f"tpad=stop_mode=clone:stop_duration={freeze_last_seconds},"
        "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p"
    )
    return [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel", "warning",
        "-framerate", str(fps),
        "-i", frames_glob_pattern,
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "20",
        "-movflags", "+faststart",
        str(output_path),
    ]


DEFAULT_EVENT_MARKERS: tuple[tuple[str, str], ...] = (
    ("2000-03-31", "2000 dot-com"),
    ("2008-09-30", "2008 GFC"),
    ("2020-03-31", "2020 COVID"),
)

DEFAULT_QUESTION_LINES: tuple[str, ...] = (
    "もし、毎月の積立タイミングを",
    "完璧に予測できたら",
    "差額は何円になるか",
)

DEFAULT_CONDITIONS: tuple[tuple[str, str, str], ...] = (
    ("A", "完璧タイミング", "各年の最安月に 年間予算を一括投資 (in hindsight)"),
    ("B", "毎月積立",     "各月に 年間予算 / 12 を等額投資"),
    ("C", "最悪タイミング", "各年の最高月に 年間予算を一括投資 (in hindsight)"),
)

DEFAULT_LIMITS: tuple[str, ...] = (
    "データ: S&P 500 Total Return（月末 close、配当再投資込）",
    "「完璧タイミング」は後知恵（look-ahead）であり実行不可",
    "税・手数料・為替・スリッページは未反映",
    "過去の結果は将来の結果を示唆しない",
)


def render_timing_race_story(
    result: pd.DataFrame,
    output_path: Path | str,
    commit_hash: str,
    summary_final_values: dict[str, float],
    invested_total: float,
    frames_dir: Path | str | None = None,
    fps: int = FRAMES_PER_SECOND_DEFAULT,
    stride: int = 1,
    race_max_frames: int | None = None,
    freeze_last_seconds: float = 1.5,
    keep_frames: bool = False,
    cold_open_seconds: float = 3.5,
    conditions_seconds: float = 3.0,
    finale_seconds: float = 4.0,
    limits_seconds: float = 3.5,
    question_lines: tuple[str, ...] = DEFAULT_QUESTION_LINES,
    conditions: tuple[tuple[str, str, str], ...] = DEFAULT_CONDITIONS,
    limits: tuple[str, ...] = DEFAULT_LIMITS,
    event_markers: tuple[tuple[str, str], ...] = DEFAULT_EVENT_MARKERS,
) -> Path:
    """v2 ストーリー版レンダラ。
    Phase 構造: [cold open] → [conditions] → [race] → [finale] → [limits] → freeze。

    既存 render_timing_race_video を破壊せず、並行して存在させる。
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")

    result = _validate_result(result)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if frames_dir is None:
        frames_dir = output_path.parent / f"{output_path.stem}_frames"
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    filename_template = "frame_{:05d}.png"

    cold_frames = max(1, int(round(cold_open_seconds * fps)))
    cond_frames = max(1, int(round(conditions_seconds * fps)))
    fin_frames = max(1, int(round(finale_seconds * fps)))
    lim_frames = max(1, int(round(limits_seconds * fps)))

    cursor = 0

    write_titlecard_frames(
        list(question_lines), frames_dir, commit_hash,
        num_frames=cold_frames, start_index=cursor,
        filename_template=filename_template,
    )
    cursor += cold_frames

    write_conditions_frames(
        list(conditions), frames_dir, commit_hash,
        num_frames=cond_frames, start_index=cursor,
        filename_template=filename_template,
    )
    cursor += cond_frames

    race_event_markers = [
        (pd.Timestamp(d), lbl) for d, lbl in event_markers
    ]
    race_frames = write_timing_race_frames(
        result=result,
        frames_dir=frames_dir,
        commit_hash=commit_hash,
        stride=stride,
        max_frames=race_max_frames,
        filename_template=filename_template,
        start_index=cursor,
        event_markers=race_event_markers,
    )
    cursor += len(race_frames)

    write_finale_frames(
        final_values=summary_final_values,
        invested_total=invested_total,
        frames_dir=frames_dir,
        commit_hash=commit_hash,
        num_frames=fin_frames,
        start_index=cursor,
        filename_template=filename_template,
    )
    cursor += fin_frames

    write_limits_frames(
        list(limits), frames_dir, commit_hash,
        num_frames=lim_frames, start_index=cursor,
        filename_template=filename_template,
    )
    cursor += lim_frames

    glob_pattern = str(frames_dir / "frame_%05d.png")
    cmd = _build_ffmpeg_cmd(glob_pattern, output_path, fps, freeze_last_seconds)
    logger.info("ffmpeg cmd: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

    if not keep_frames:
        for p in frames_dir.glob("frame_*.png"):
            p.unlink(missing_ok=True)
        try:
            frames_dir.rmdir()
        except OSError:
            logger.debug("frames_dir not empty after cleanup: %s", frames_dir)

    return output_path


def render_timing_race_video(
    result: pd.DataFrame,
    output_path: Path | str,
    commit_hash: str,
    frames_dir: Path | str | None = None,
    fps: int = FRAMES_PER_SECOND_DEFAULT,
    stride: int = 1,
    max_frames: int | None = None,
    freeze_last_seconds: float = 2.0,
    keep_frames: bool = False,
) -> Path:
    """result DataFrame から mp4 を 1 本生成する。

    Args:
        result: simulate_timing_race の出力
        output_path: 出力 mp4 のパス
        commit_hash: コミットハッシュ字幕
        frames_dir: フレーム保存先（None なら output_path と同じフォルダ下の frames/）
        fps: フレームレート
        stride: N 行に 1 フレーム（例: 1 = 月次、12 = 年次）
        max_frames: 上限（スモークテスト用）
        freeze_last_seconds: 末尾フレームの追加ホールド秒数
        keep_frames: False なら完了後に frames_dir の PNG を削除

    Returns:
        生成された mp4 の Path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if frames_dir is None:
        frames_dir = output_path.parent / f"{output_path.stem}_frames"
    frames_dir = Path(frames_dir)

    filename_template = "frame_{:05d}.png"
    write_timing_race_frames(
        result=result,
        frames_dir=frames_dir,
        commit_hash=commit_hash,
        stride=stride,
        max_frames=max_frames,
        filename_template=filename_template,
    )

    glob_pattern = str(frames_dir / "frame_%05d.png")
    cmd = _build_ffmpeg_cmd(glob_pattern, output_path, fps, freeze_last_seconds)
    logger.info("ffmpeg cmd: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

    if not keep_frames:
        for p in frames_dir.glob("frame_*.png"):
            p.unlink(missing_ok=True)
        try:
            frames_dir.rmdir()
        except OSError:
            logger.debug("frames_dir not empty after cleanup: %s", frames_dir)

    return output_path
