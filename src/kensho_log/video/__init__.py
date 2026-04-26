"""動画生成レイヤ。

- matplotlib でフレーム PNG を 1 枚ずつ生成（メモリ常駐を最小化）
- FFmpeg を subprocess 直接呼び出しで結合（MoviePy 等の重量ライブラリは使用禁止）

8GB RAM 環境を前提に以下を徹底:
- フレームごとに figure を close してメモリを開放
- 画素は 1280x720 を基本、拡張時も 1920x1080 まで
"""

from .story_v3 import render_timing_race_story_v3
from .timing_race_animation import (
    DEFAULT_CONDITIONS,
    DEFAULT_EVENT_MARKERS,
    DEFAULT_LIMITS,
    DEFAULT_QUESTION_LINES,
    FRAMES_PER_SECOND_DEFAULT,
    FfmpegNotFoundError,
    race_frame_indices,
    render_timing_race_story,
    render_timing_race_video,
    write_conditions_frames,
    write_finale_frames,
    write_limits_frames,
    write_timing_race_frames,
    write_titlecard_frames,
)
from .voicevox import is_engine_up, synthesize_wav

__all__ = [
    "DEFAULT_CONDITIONS",
    "DEFAULT_EVENT_MARKERS",
    "DEFAULT_LIMITS",
    "DEFAULT_QUESTION_LINES",
    "FRAMES_PER_SECOND_DEFAULT",
    "FfmpegNotFoundError",
    "is_engine_up",
    "race_frame_indices",
    "render_timing_race_story",
    "render_timing_race_story_v3",
    "render_timing_race_video",
    "synthesize_wav",
    "write_conditions_frames",
    "write_finale_frames",
    "write_limits_frames",
    "write_timing_race_frames",
    "write_titlecard_frames",
]
