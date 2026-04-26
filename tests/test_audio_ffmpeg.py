"""audio_ffmpeg のテスト（subprocess は monkeypatch でモック）."""

from __future__ import annotations

from pathlib import Path

import pytest

from kensho_log.video import audio_ffmpeg as af


class _Recorder:
    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, cmd, check=False, capture_output=False, text=False):
        self.calls.append(list(cmd))
        class _R:
            stdout = "1.234"
            returncode = 0
        return _R()


def test_concat_wavs_empty_raises(tmp_path: Path):
    with pytest.raises(ValueError):
        af.concat_wavs_to_wav([], tmp_path / "out.wav")


def test_concat_wavs_single_uses_simple_input(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(af.shutil, "which", lambda _: "ffmpeg")
    rec = _Recorder()
    monkeypatch.setattr(af.subprocess, "run", rec)
    in_wav = tmp_path / "a.wav"
    in_wav.write_bytes(b"RIFF")
    out_wav = tmp_path / "out.wav"
    af.concat_wavs_to_wav([in_wav], out_wav)
    assert len(rec.calls) == 1
    cmd = rec.calls[0]
    assert "concat" not in cmd
    assert str(in_wav) in cmd


def test_concat_wavs_multi_uses_concat_demuxer(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(af.shutil, "which", lambda _: "ffmpeg")
    rec = _Recorder()
    monkeypatch.setattr(af.subprocess, "run", rec)
    a = tmp_path / "a.wav"; a.write_bytes(b"a")
    b = tmp_path / "b.wav"; b.write_bytes(b"b")
    out = tmp_path / "out.wav"
    af.concat_wavs_to_wav([a, b], out)
    assert len(rec.calls) == 1
    cmd = rec.calls[0]
    assert "-f" in cmd and "concat" in cmd
    assert "-safe" in cmd and "0" in cmd


def test_media_duration_parses(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(af.shutil, "which", lambda _: "ffprobe")
    rec = _Recorder()
    monkeypatch.setattr(af.subprocess, "run", rec)
    p = tmp_path / "x.wav"; p.write_bytes(b"x")
    d = af.media_duration(p)
    assert d == pytest.approx(1.234)
