"""仮説 001: v3（3 ペイン + 台本テロップ + VOICEVOX 音声、フェーズ別 sync）.

設計:
- 各フェーズの台本を別個に TTS → WAV を保存し、ffprobe で長さを取得
- その秒数を「映像側のそのフェーズの長さ」として採用（fps を掛けてフレーム数）
- レースは num_frames モードで一様サンプリング
- 全フェーズの WAV を ffmpeg concat で 1 本に → 無声 mp4 と mux

無声モード（``synthesize_audio=False`` または ``KENSHO_TTS=0``）の場合は
旧来の固定秒（``cold_open_seconds`` 等）でフレームを生成する。
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from . import timing_race_animation as tra
from .audio_ffmpeg import (
    adjust_wav_duration,
    concat_wavs_to_wav,
    media_duration,
    mux_video_and_audio,
)
from .narration_001 import (
    NARRATION_COLD,
    NARRATION_CONDITIONS,
    NARRATION_FINALE,
    NARRATION_LIMITS,
    NARRATION_RACE,
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


@dataclass(frozen=True)
class _PhaseSpec:
    name: str
    text: str
    fallback_seconds: float


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


def _synthesize_phase_wav(
    text: str, out_dir: Path, name: str, speaker: int, base_url: str
) -> Path:
    raw = synthesize_wav(text, speaker=speaker, base_url=base_url)
    p = out_dir / f"phase_{name}.wav"
    p.write_bytes(raw)
    return p


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
    race_seconds_no_audio: float | None = None,
    question_lines: tuple[str, ...] = tra.DEFAULT_QUESTION_LINES,
    conditions: tuple[tuple[str, str, str], ...] = tra.DEFAULT_CONDITIONS,
    limits: tuple[str, ...] = tra.DEFAULT_LIMITS,
    event_markers: tuple[tuple[str, str], ...] = tra.DEFAULT_EVENT_MARKERS,
    *,
    synthesize_audio: bool = True,
    voicevox_speaker: int = DEFAULT_SPEAKER,
    voicevox_base: str = DEFAULT_BASE,
    tts_text: str = TTS_FULL_DEFAULT,  # 互換用、フェーズ別を使う場合は無視
) -> dict[str, Any]:
    """3 ペイン + 下テロップ + フェーズ別音声同期。

    返却: ``{"video": str, "audio_synthesized": bool, "phase_durations": dict[str, float]}``
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    _ = tts_text  # 旧 API 互換のため受けるだけ
    if os.environ.get("KENSHO_TTS", "").strip() in ("0", "false", "False"):
        synthesize_audio = False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if frames_dir is None:
        frames_dir = output_path.parent / f"{output_path.stem}_frames"
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    phases = [
        _PhaseSpec("cold", NARRATION_COLD, cold_open_seconds),
        _PhaseSpec("conditions", NARRATION_CONDITIONS, conditions_seconds),
        _PhaseSpec("race", NARRATION_RACE, race_seconds_no_audio
                   if race_seconds_no_audio is not None else 14.0),
        _PhaseSpec("finale", NARRATION_FINALE, finale_seconds),
        _PhaseSpec("limits", NARRATION_LIMITS, limits_seconds),
    ]

    audio_dir: Path | None = None
    durations: dict[str, float] = {}
    audio_files: dict[str, Path] = {}

    if synthesize_audio:
        if not is_engine_up(voicevox_base):
            raise VoicevoxError(
                f"VOICEVOX Engine が {voicevox_base} に応答しません。"
                "起動するか、--no-audio か KENSHO_TTS=0 を指定してください。"
            )
        audio_dir = output_path.parent / f"{output_path.stem}_audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        for ph in phases:
            wav = _synthesize_phase_wav(
                ph.text, audio_dir, ph.name, voicevox_speaker, voicevox_base
            )
            d = media_duration(wav)
            audio_files[ph.name] = wav
            durations[ph.name] = d
            logger.info("phase=%s wav=%s duration=%.3fs", ph.name, wav.name, d)
    else:
        for ph in phases:
            durations[ph.name] = float(ph.fallback_seconds)

    filename_template = "frame_{:05d}.png"
    cur = 0

    cold_f = max(1, int(round(durations["cold"] * fps)))
    cond_f = max(1, int(round(durations["conditions"] * fps)))
    race_f = max(1, int(round(durations["race"] * fps)))
    fin_f = max(1, int(round(durations["finale"] * fps)))
    lim_f = max(1, int(round(durations["limits"] * fps)))

    placeholder = "log  (CG placeholder)"

    tra.write_titlecard_frames(
        list(question_lines), frames_dir, commit_hash, num_frames=cold_f,
        start_index=cur, filename_template=filename_template,
        three_pane=True, bottom_telop=TELP_COLD, placeholder_label=placeholder,
    )
    cur += cold_f
    tra.write_conditions_frames(
        list(conditions), frames_dir, commit_hash, num_frames=cond_f,
        start_index=cur, filename_template=filename_template,
        three_pane=True, bottom_telop=TELP_CONDITIONS, placeholder_label=placeholder,
    )
    cur += cond_f
    tra.write_timing_race_frames(
        result=result, frames_dir=frames_dir, commit_hash=commit_hash,
        filename_template=filename_template, start_index=cur,
        event_markers=[(pd.Timestamp(d), lbl) for d, lbl in event_markers],
        three_pane=True,
        telop_by_frame=build_race_telop_lines(race_f),
        placeholder_label=placeholder,
        num_frames=race_f,
    )
    cur += race_f
    tra.write_finale_frames(
        final_values=summary_final_values, invested_total=invested_total,
        frames_dir=frames_dir, commit_hash=commit_hash, num_frames=fin_f, start_index=cur,
        filename_template=filename_template,
        three_pane=True, bottom_telop=TELP_FINALE, placeholder_label=placeholder,
    )
    cur += fin_f
    tra.write_limits_frames(
        list(limits), frames_dir, commit_hash, num_frames=lim_f, start_index=cur,
        filename_template=filename_template,
        three_pane=True, bottom_telop=TELP_LIMITS, placeholder_label=placeholder,
    )

    work_silent = output_path.parent / f"_{output_path.stem}_silent_work.mp4"
    glob_p = str(frames_dir / "frame_%05d.png")
    _encode_silent_ffmpeg(glob_p, work_silent, fps, freeze_last_seconds)

    if not synthesize_audio:
        if work_silent != output_path:
            work_silent.replace(output_path)
        if not keep_frames:
            _cleanup_frame_dir(frames_dir)
        return {
            "video": str(output_path),
            "audio_synthesized": False,
            "phase_durations": durations,
        }

    assert audio_dir is not None
    with tempfile.TemporaryDirectory() as tdir:
        tdir_p = Path(tdir)
        wav_concat = tdir_p / "narration_concat.wav"
        concat_wavs_to_wav(
            [audio_files[ph.name] for ph in phases], wav_concat
        )
        d_v = media_duration(work_silent)
        adj = tdir_p / "narration_adj.wav"
        adjust_wav_duration(wav_concat, d_v, adj)
        mux_video_and_audio(work_silent, adj, output_path)

    if work_silent.exists():
        work_silent.unlink(missing_ok=True)
    if not keep_frames:
        _cleanup_frame_dir(frames_dir)
        if audio_dir is not None and audio_dir.is_dir():
            for p in audio_dir.glob("phase_*.wav"):
                p.unlink(missing_ok=True)
            try:
                audio_dir.rmdir()
            except OSError:
                pass

    return {
        "video": str(output_path),
        "audio_synthesized": True,
        "phase_durations": durations,
    }
