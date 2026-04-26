"""仮説 001 Timing Race の mp4 を FFmpeg で生成する。

使い方:
    python scripts/render_001.py                    # v1: レースのみ
    python scripts/render_001.py --v2               # v2: 疑問提示→条件→レース→フィナーレ→限界
    python scripts/render_001.py --smoke            # 先頭 24 フレームのみの動作確認
    python scripts/render_001.py --stride 3         # 3ヶ月に1フレーム

前提:
    - scripts/run_001.py を先に実行し
      output/tmp/001_timing_race_monthly.csv が存在すること
    - FFmpeg が PATH 上にあること
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import pandas as pd  # noqa: E402

from kensho_log.video import (  # noqa: E402
    FRAMES_PER_SECOND_DEFAULT,
    FfmpegNotFoundError,
    render_timing_race_story,
    render_timing_race_video,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("render_001")

INPUT_CSV = REPO_ROOT / "output" / "tmp" / "001_timing_race_monthly.csv"
OUTPUT_MP4 = REPO_ROOT / "output" / "videos" / "001_timing_race.mp4"


def _get_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _load_result(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. run `python scripts/run_001.py` first."
        )
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df.index.name = df.index.name or "month_end"
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Render hypothesis 001 timing race mp4")
    parser.add_argument(
        "--stride", type=int, default=1,
        help="N-row stride (1=every month, 3=every 3 months)",
    )
    parser.add_argument(
        "--fps", type=int, default=FRAMES_PER_SECOND_DEFAULT,
        help="output framerate",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="limit to first 24 frames for a quick sanity check",
    )
    parser.add_argument(
        "--keep-frames", action="store_true",
        help="keep intermediate PNG frames after encoding",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_MP4,
        help="output mp4 path",
    )
    parser.add_argument(
        "--v2", action="store_true",
        help="render v2 story version (cold open + conditions + race + finale + limits)",
    )
    args = parser.parse_args()

    result = _load_result(INPUT_CSV)
    commit = _get_commit_hash()

    logger.info(
        "input=%s rows=%d stride=%d fps=%d smoke=%s v2=%s commit=%s",
        INPUT_CSV.relative_to(REPO_ROOT), len(result), args.stride, args.fps,
        args.smoke, args.v2, commit[:12],
    )

    max_frames = 24 if args.smoke else None
    output = args.output
    if not output.is_absolute():
        output = (REPO_ROOT / output).resolve()
    if args.v2 and output == OUTPUT_MP4:
        output = output.with_name("001_timing_race_v2.mp4")
    if args.smoke and output.name.endswith(".mp4"):
        output = output.with_name(output.stem + "_smoke.mp4")

    try:
        if args.v2:
            final_a = float(result["A_value"].iloc[-1])
            final_b = float(result["B_value"].iloc[-1])
            final_c = float(result["C_value"].iloc[-1])
            invested_total = float(result["A_invested_cum"].iloc[-1])
            written = render_timing_race_story(
                result=result,
                output_path=output,
                commit_hash=commit,
                summary_final_values={"A": final_a, "B": final_b, "C": final_c},
                invested_total=invested_total,
                fps=args.fps,
                stride=args.stride,
                race_max_frames=max_frames,
                freeze_last_seconds=1.5,
                keep_frames=args.keep_frames,
                cold_open_seconds=0.5 if args.smoke else 3.5,
                conditions_seconds=0.3 if args.smoke else 3.0,
                finale_seconds=0.5 if args.smoke else 4.0,
                limits_seconds=0.3 if args.smoke else 3.5,
            )
        else:
            written = render_timing_race_video(
                result=result,
                output_path=output,
                commit_hash=commit,
                fps=args.fps,
                stride=args.stride,
                max_frames=max_frames,
                freeze_last_seconds=2.0,
                keep_frames=args.keep_frames,
            )
    except FfmpegNotFoundError as exc:
        logger.error("ffmpeg missing: %s", exc)
        return 2

    size_kb = written.stat().st_size / 1024
    try:
        display_path = written.relative_to(REPO_ROOT)
    except ValueError:
        display_path = written
    logger.info("wrote %s (%.1f KB)", display_path, size_kb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
