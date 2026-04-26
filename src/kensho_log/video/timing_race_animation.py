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

    Returns:
        生成されたフレームの Path リスト（昇順）
    """
    if stride < 1:
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

    idxs = list(range(0, len(result), stride))
    if idxs[-1] != len(result) - 1:
        idxs.append(len(result) - 1)
    if max_frames is not None and max_frames > 0:
        idxs = idxs[:max_frames]

    produced: list[Path] = []
    for fi, row_i in enumerate(idxs):
        slice_end = row_i + 1
        visible = result.iloc[:slice_end]

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
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
        box_x = 0.98
        box_y = 0.42
        line_h = 0.06

        for i, (label, value, color) in enumerate(
            [
                ("A", last["A_value"], COLOR_A),
                ("B", last["B_value"], COLOR_B),
                ("C", last["C_value"], COLOR_C),
            ]
        ):
            fig.text(
                box_x - 0.16, box_y - i * line_h, label,
                ha="right", va="center", fontsize=12, color=color,
                fontproperties=fonts.mincho,
            )
            fig.text(
                box_x, box_y - i * line_h, _format_jpy(value),
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

        ax.get_legend().remove() if ax.get_legend() else None

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

        frame_path = frames_dir / filename_template.format(fi)
        fig.savefig(frame_path, dpi=dpi)
        plt.close(fig)
        produced.append(frame_path)

    logger.info("wrote %d frames to %s", len(produced), frames_dir)
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
