"""動画生成レイヤ。

- matplotlib でフレーム PNG を 1 枚ずつ生成（メモリ常駐を最小化）
- FFmpeg を subprocess 直接呼び出しで結合（MoviePy 等の重量ライブラリは使用禁止）

8GB RAM 環境を前提に以下を徹底:
- フレームごとに figure を close してメモリを開放
- 画素は 1280x720 を基本、拡張時も 1920x1080 まで
"""

from .timing_race_animation import (
    FRAMES_PER_SECOND_DEFAULT,
    FfmpegNotFoundError,
    render_timing_race_video,
    write_timing_race_frames,
)

__all__ = [
    "FRAMES_PER_SECOND_DEFAULT",
    "FfmpegNotFoundError",
    "render_timing_race_video",
    "write_timing_race_frames",
]
