"""video.timing_race_animation のテスト。

- ffmpeg 実バイナリを要求するテストはマーカー "integration" を付け
  デフォルトでスキップ（CI で ffmpeg を個別用意する判断を後回しにするため）。
- フレーム PNG 生成のみオフラインで検証する。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kensho_log.video import timing_race_animation as tra


def _sample_result(n: int = 6) -> pd.DataFrame:
    idx = pd.date_range("2020-01-31", periods=n, freq="ME")
    close = np.linspace(100.0, 200.0, n)
    invested = np.linspace(0, 1_200_000, n)
    a_val = invested * 1.10
    b_val = invested * 1.05
    c_val = invested * 1.00
    return pd.DataFrame(
        {
            "close": close,
            "year": idx.year,
            "A_invested_cum": invested,
            "B_invested_cum": invested,
            "C_invested_cum": invested,
            "A_units_cum": a_val / close,
            "B_units_cum": b_val / close,
            "C_units_cum": c_val / close,
            "A_value": a_val,
            "B_value": b_val,
            "C_value": c_val,
        },
        index=idx,
    )


def _is_png(path: Path) -> bool:
    with path.open("rb") as f:
        return f.read(8) == b"\x89PNG\r\n\x1a\n"


class TestWriteFrames:
    def test_happy_path_writes_all_frames(self, tmp_path):
        result = _sample_result(6)
        produced = tra.write_timing_race_frames(
            result=result,
            frames_dir=tmp_path,
            commit_hash="0123456789abcdef" * 2,
        )
        assert len(produced) == 6
        for p in produced:
            assert p.exists()
            assert p.stat().st_size > 0
            assert _is_png(p)

    def test_stride_reduces_frame_count_but_keeps_last(self, tmp_path):
        result = _sample_result(10)
        produced = tra.write_timing_race_frames(
            result=result,
            frames_dir=tmp_path,
            commit_hash="deadbeef" * 5,
            stride=3,
        )
        assert 0 < len(produced) < 10
        assert produced[-1].exists()

    def test_max_frames_respected(self, tmp_path):
        result = _sample_result(8)
        produced = tra.write_timing_race_frames(
            result=result,
            frames_dir=tmp_path,
            commit_hash="abc123" * 8,
            max_frames=3,
        )
        assert len(produced) == 3

    def test_stride_zero_raises(self, tmp_path):
        result = _sample_result(3)
        with pytest.raises(ValueError):
            tra.write_timing_race_frames(
                result=result,
                frames_dir=tmp_path,
                commit_hash="x" * 40,
                stride=0,
            )

    def test_missing_columns_raises(self, tmp_path):
        idx = pd.date_range("2020-01-31", periods=3, freq="ME")
        bad = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)
        with pytest.raises(ValueError):
            tra.write_timing_race_frames(
                result=bad,
                frames_dir=tmp_path,
                commit_hash="y" * 40,
            )

    def test_empty_raises(self, tmp_path):
        idx = pd.DatetimeIndex([])
        cols = {
            "close": [], "A_value": [], "B_value": [], "C_value": [],
            "A_invested_cum": [], "B_invested_cum": [], "C_invested_cum": [],
        }
        with pytest.raises(ValueError):
            tra.write_timing_race_frames(
                result=pd.DataFrame(cols, index=idx),
                frames_dir=tmp_path,
                commit_hash="z" * 40,
            )

    def test_non_datetime_index_raises(self, tmp_path):
        cols = {
            "close": [1.0, 2.0], "A_value": [1.0, 2.0], "B_value": [1.0, 2.0],
            "C_value": [1.0, 2.0],
            "A_invested_cum": [1.0, 2.0], "B_invested_cum": [1.0, 2.0],
            "C_invested_cum": [1.0, 2.0],
        }
        with pytest.raises(TypeError):
            tra.write_timing_race_frames(
                result=pd.DataFrame(cols, index=[0, 1]),
                frames_dir=tmp_path,
                commit_hash="w" * 40,
            )


class TestFfmpegCommandBuilder:
    def test_command_includes_input_glob_and_output(self, monkeypatch):
        monkeypatch.setattr(tra, "_ffmpeg_binary", lambda: "ffmpeg")
        cmd = tra._build_ffmpeg_cmd(
            frames_glob_pattern="/tmp/frame_%05d.png",
            output_path=Path("/tmp/out.mp4"),
            fps=30,
            freeze_last_seconds=2.0,
        )
        assert cmd[0] == "ffmpeg"
        assert "-framerate" in cmd
        assert "30" in cmd
        assert "/tmp/frame_%05d.png" in cmd
        assert cmd[-1].replace("\\", "/").endswith("/tmp/out.mp4") or cmd[-1].endswith("out.mp4")
        assert "-c:v" in cmd and "libx264" in cmd
        vf_index = cmd.index("-vf")
        vf_value = cmd[vf_index + 1]
        assert "tpad=stop_mode=clone:stop_duration=2.0" in vf_value
        assert "yuv420p" in vf_value

    def test_ffmpeg_not_on_path_raises(self, monkeypatch):
        monkeypatch.setattr(tra.shutil, "which", lambda _: None)
        with pytest.raises(tra.FfmpegNotFoundError):
            tra._ffmpeg_binary()
