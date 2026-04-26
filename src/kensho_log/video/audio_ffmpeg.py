"""ffprobe による秒数取得と、WAV 長のパディング / トリム、映像＋音声のマルチプレックス."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


def _ffmpeg() -> str:
    p = shutil.which("ffmpeg")
    if p is None:
        raise FileNotFoundError("ffmpeg not on PATH")
    return p


def _ffprobe() -> str:
    p = shutil.which("ffprobe")
    if p is None:
        raise FileNotFoundError("ffprobe not on PATH")
    return p


def media_duration(path: Path | str) -> float:
    """動画 or 音声の再生時間（秒）。"""
    p = str(path)
    r = subprocess.run(
        [
            _ffprobe(), "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", p,
        ],
        check=True, capture_output=True, text=True,
    )
    v = (r.stdout or "0").strip() or "0"
    return float(v)


def adjust_wav_duration(
    wav_in: Path | str, target_seconds: float, wav_out: Path | str
) -> Path:
    """WAV（または任意 FFmpeg が読む音声）を target_seconds 秒に合わせる。

    - 短い: 無音 apad
    - 長い: atrim
    サンプルレート等は入出力に任せる（再エンコードあり）。
    """
    inp, out = str(wav_in), str(wav_out)
    t = max(0.05, float(target_seconds))
    d_in = media_duration(inp)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    if abs(d_in - t) < 0.02:
        import shutil
        shutil.copy2(inp, out)
        return Path(out)
    if d_in > t + 0.02:
        cmd = [
            _ffmpeg(), "-y", "-hide_banner", "-loglevel", "warning", "-i", inp,
            "-t", f"{t:.3f}", "-c:a", "pcm_s16le", out,
        ]
        subprocess.run(cmd, check=True)
        return Path(out)
    pad = t - d_in
    cmd = [
        _ffmpeg(), "-y", "-hide_banner", "-loglevel", "warning", "-i", inp,
        "-af", f"apad=pad_dur={pad}", "-t", f"{t:.3f}", "-c:a", "pcm_s16le", out,
    ]
    subprocess.run(cmd, check=True)
    return Path(out)


def concat_wavs_to_wav(
    inputs: list[Path | str], out_wav: Path | str
) -> Path:
    """複数 WAV を 1 本に連結（同一サンプルレート/チャンネル前提、同じでなければ ffmpeg が再エンコード）.

    実装: ffmpeg concat demuxer を使う。listfile を一時ファイルとして書く。
    """
    if not inputs:
        raise ValueError("inputs must not be empty")
    out = Path(out_wav)
    out.parent.mkdir(parents=True, exist_ok=True)

    if len(inputs) == 1:
        cmd = [
            _ffmpeg(), "-y", "-hide_banner", "-loglevel", "warning",
            "-i", str(inputs[0]), "-c:a", "pcm_s16le", str(out),
        ]
        subprocess.run(cmd, check=True)
        return out

    import tempfile
    fd, tname = tempfile.mkstemp(prefix="kensho_concat_", suffix=".txt")
    list_path = Path(tname)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        for p in inputs:
            ap = Path(p).resolve()
            esc = str(ap).replace("\\", "/").replace("'", r"'\''")
            f.write(f"file '{esc}'\n")
    try:
        cmd = [
            _ffmpeg(), "-y", "-hide_banner", "-loglevel", "warning",
            "-f", "concat", "-safe", "0",
            "-i", str(list_path),
            "-c:a", "pcm_s16le", str(out),
        ]
        logger.info("ffmpeg concat: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)
    finally:
        list_path.unlink(missing_ok=True)
    return out


def mux_video_and_audio(
    video_path: Path | str, audio_wav: Path | str, out_path: Path | str
) -> Path:
    """映像（コピー）+ モノラル AAC を 1 本の mp4 に。長さは映像基準（音声は揃え済み想定）。"""
    v, a, o = str(video_path), str(audio_wav), str(out_path)
    cmd = [
        _ffmpeg(), "-y", "-hide_banner", "-loglevel", "warning",
        "-i", v, "-i", a,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "1",
        o,
    ]
    logger.info("ffmpeg mux: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return Path(out_path)
