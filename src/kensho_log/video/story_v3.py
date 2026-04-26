"""仮説 001: v3（3 ペイン + 台本テロップ + VOICEVOX 音声、任意）."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from . import timing_race_animation as tra
from .audio_ffmpeg import adjust_wav_duration, media_duration, mux_video_and_audio
from .narration_001 import (
    TELP_COLD,
    TELP_CONDITIONS,
    TELP_FINALE,
    TELP_LIMITS,
    TTS_FULL_DEFAULT,
    build_race_telop_lines,
)
from .voicevox import (
    DEFAULT_BASE,
    DEFAULT_SPEAKER,
    VoicevoxError,
    is_engine_up,
    synthesize_wav,
)

logger = logging.getLogger(__name__)


def _ffmpeg_binary() -> str:
    p = shutil.which("ffmpeg")
    if p is None:
        raise tra.FfmpegNotFoundError(
            "ffmpeg not found on PATH. install ffmpeg and ensure it is on PATH."
        )
    return p


def _encode_silent_ffmpeg(
    glob_pattern: str, output_path: Path, fps: int, freeze_last_seconds: float
) -> None:
    vf = (
        f"tpad=stop_mode=clone:stop_duration={freeze_last_seconds},"
        "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p"
    )
    cmd = [
        _ffmpeg_binary(), "-y", "-hide_banner", "-loglevel", "warning",
        "-framerate", str(fps), "-i", glob_pattern, "-vf", vf,
        "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-movflags", "+faststart",
        str(output_path),
    ]
    logger.info("ffmpeg (silent) cmd: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _cleanup_frame_dir(frames_dir: Path) -> None:
    if not frames_dir.is_dir():
        return
    for p in frames_dir.glob("frame_*.png"):
        p.unlink(missing_ok=True)
    try:
        frames_dir.rmdir()
    except OSError:
        pass


def render_timing_race_story_v3(
    result: pd.DataFrame,
    output_path: Path | str,
    commit_hash: str,
    summary_final_values: dict[str, float],
    invested_total: float,
    frames_dir: Path | str | None = None,
    fps: int = tra.FRAMES_PER_SECOND_DEFAULT,
    stride: int = 1,
    race_max_frames: int | None = None,
    freeze_last_seconds: float = 1.5,
    keep_frames: bool = False,
    cold_open_seconds: float = 3.5,
    conditions_seconds: float = 3.0,
    finale_seconds: float = 4.0,
    limits_seconds: float = 3.5,
    question_lines: tuple[str, ...] = tra.DEFAULT_QUESTION_LINES,
    conditions: tuple[tuple[str, str, str], ...] = tra.DEFAULT_CONDITIONS,
    limits: tuple[str, ...] = tra.DEFAULT_LIMITS,
    event_markers: tuple[tuple[str, str], ...] = tra.DEFAULT_EVENT_MARKERS,
    *,
    synthesize_audio: bool = True,
    tts_text: str = TTS_FULL_DEFAULT,
    voicevox_speaker: int = DEFAULT_SPEAKER,
    voicevox_base: str = DEFAULT_BASE,
    tts_wav_path: Path | str | None = None,
) -> dict[str, Any]:
    """3 ペイン + 下テロップ。音声は ``synthesize_audio`` と Engine 有無に依存。

    返却例: ``{"video": "path/to/out.mp4", "audio_synthesized": true}``
    環境変数 ``KENSHO_TTS=0`` で TTS 無効（無声 mp4 のみ）。

    無声部分は ``<stem>_silent_work.mp4`` を経由して最終的に上書きまたは削除。
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if os.environ.get("KENSHO_TTS", "").strip() in ("0", "false", "False"):
        synthesize_audio = False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if frames_dir is None:
        frames_dir = output_path.parent / f"{output_path.stem}_frames"
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    filename_template = "frame_{:05d}.png"
    cold_f = max(1, int(round(cold_open_seconds * fps)))
    cond_f = max(1, int(round(conditions_seconds * fps)))
    fin_f = max(1, int(round(finale_seconds * fps)))
    lim_f = max(1, int(round(limits_seconds * fps)))
    cur = 0

    tra.write_titlecard_frames(
        list(question_lines), frames_dir, commit_hash, num_frames=cold_f,
        start_index=cur, filename_template=filename_template,
        three_pane=True, bottom_telop=TELP_COLD, placeholder_label="log  (CG placeholder)",
    )
    cur += cold_f
    tra.write_conditions_frames(
        list(conditions), frames_dir, commit_hash, num_frames=cond_f,
        start_index=cur, filename_template=filename_template,
        three_pane=True, bottom_telop=TELP_CONDITIONS, placeholder_label="log  (CG placeholder)",
    )
    cur += cond_f
    n_race = len(tra.race_frame_indices(len(result), stride, race_max_frames))
    tra.write_timing_race_frames(
        result=result, frames_dir=frames_dir, commit_hash=commit_hash,
        stride=stride, max_frames=race_max_frames, filename_template=filename_template,
        start_index=cur, event_markers=[(pd.Timestamp(d), lbl) for d, lbl in event_markers],
        three_pane=True,
        telop_by_frame=build_race_telop_lines(n_race),
        placeholder_label="log  (CG placeholder)",
    )
    cur += n_race
    tra.write_finale_frames(
        final_values=summary_final_values, invested_total=invested_total,
        frames_dir=frames_dir, commit_hash=commit_hash, num_frames=fin_f, start_index=cur,
        filename_template=filename_template,
        three_pane=True, bottom_telop=TELP_FINALE, placeholder_label="log  (CG placeholder)",
    )
    cur += fin_f
    tra.write_limits_frames(
        list(limits), frames_dir, commit_hash, num_frames=lim_f, start_index=cur,
        filename_template=filename_template,
        three_pane=True, bottom_telop=TELP_LIMITS, placeholder_label="log  (CG placeholder)",
    )
    _ = cur  # 検証: 上記と整合

    work_silent = output_path.parent / f"_{output_path.stem}_silent_work.mp4"
    glob_p = str(frames_dir / "frame_%05d.png")
    _encode_silent_ffmpeg(glob_p, work_silent, fps, freeze_last_seconds)

    ext_wav = Path(tts_wav_path) if tts_wav_path else None
    has_file = ext_wav is not None and ext_wav.is_file()
    if not has_file and not synthesize_audio:
        if work_silent != output_path:
            work_silent.replace(output_path)
        if not keep_frames:
            _cleanup_frame_dir(frames_dir)
        return {"video": str(output_path), "audio_synthesized": False}

    if has_file:
        use_path: Path = ext_wav
        made_temp = False
    else:
        if not is_engine_up(voicevox_base):
            work_silent.unlink(missing_ok=True)
            if not keep_frames:
                _cleanup_frame_dir(frames_dir)
            raise VoicevoxError(
                f"VOICEVOX Engine が {voicevox_base} に応答しません。"
                "起動するか、--no-audio か KENSHO_TTS=0、または --tts-wav の wav を渡してください。"
            )
        raw = synthesize_wav(
            tts_text, speaker=voicevox_speaker, base_url=voicevox_base,
        )
        fd, tname = tempfile.mkstemp(
            prefix="kensho_tts_", suffix=".wav", dir=str(output_path.parent)
        )
        os.close(fd)
        use_path = Path(tname)
        use_path.write_bytes(raw)
        made_temp = True

    d_v = media_duration(work_silent)
    with tempfile.TemporaryDirectory() as tdir:
        tdir_p = Path(tdir)
        adj = tdir_p / "adj.wav"
        adjust_wav_duration(use_path, d_v, adj)
        mux_video_and_audio(work_silent, adj, output_path)
    if made_temp and use_path.exists():
        try:
            use_path.unlink()
        except OSError:
            pass
    if work_silent.exists():
        work_silent.unlink(missing_ok=True)
    if not keep_frames:
        _cleanup_frame_dir(frames_dir)
    return {"video": str(output_path), "audio_synthesized": True}
