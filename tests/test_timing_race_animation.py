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


class TestTitleCard:
    def test_writes_num_frames(self, tmp_path):
        produced = tra.write_titlecard_frames(
            ["line1", "line2"], tmp_path, "c0ffee" * 7,
            num_frames=5, fade_in_frames=2,
        )
        assert len(produced) == 5
        for p in produced:
            assert p.exists()
            assert _is_png(p)

    def test_start_index_offsets_filenames(self, tmp_path):
        produced = tra.write_titlecard_frames(
            ["hello"], tmp_path, "deadbeef" * 5,
            num_frames=3, start_index=100, fade_in_frames=0,
        )
        names = sorted(p.name for p in produced)
        assert names == ["frame_00100.png", "frame_00101.png", "frame_00102.png"]

    def test_zero_frames_raises(self, tmp_path):
        with pytest.raises(ValueError):
            tra.write_titlecard_frames(["x"], tmp_path, "h" * 40, num_frames=0)

    def test_negative_fade_raises(self, tmp_path):
        with pytest.raises(ValueError):
            tra.write_titlecard_frames(
                ["x"], tmp_path, "h" * 40, num_frames=5, fade_in_frames=-1,
            )


class TestConditionsCard:
    def test_writes_num_frames(self, tmp_path):
        conditions = [
            ("A", "label_a", "desc_a"),
            ("B", "label_b", "desc_b"),
            ("C", "label_c", "desc_c"),
        ]
        produced = tra.write_conditions_frames(
            conditions, tmp_path, "abcdef" * 7, num_frames=4,
        )
        assert len(produced) == 4
        assert all(p.exists() and _is_png(p) for p in produced)

    def test_empty_conditions_raises(self, tmp_path):
        with pytest.raises(ValueError):
            tra.write_conditions_frames([], tmp_path, "x" * 40, num_frames=3)


class TestFinale:
    def test_writes_num_frames(self, tmp_path):
        produced = tra.write_finale_frames(
            final_values={"A": 10_000_000.0, "B": 8_000_000.0, "C": 6_000_000.0},
            invested_total=5_000_000.0,
            frames_dir=tmp_path,
            commit_hash="f00d" * 10,
            num_frames=6,
        )
        assert len(produced) == 6
        assert all(_is_png(p) for p in produced)

    def test_bad_keys_raises(self, tmp_path):
        with pytest.raises(ValueError):
            tra.write_finale_frames(
                final_values={"X": 1.0, "Y": 2.0},
                invested_total=1.0,
                frames_dir=tmp_path,
                commit_hash="x" * 40,
                num_frames=3,
            )

    def test_bad_odometer_ratio_raises(self, tmp_path):
        with pytest.raises(ValueError):
            tra.write_finale_frames(
                final_values={"A": 1.0, "B": 1.0, "C": 1.0},
                invested_total=1.0,
                frames_dir=tmp_path,
                commit_hash="x" * 40,
                num_frames=3,
                odometer_ratio=0.0,
            )


class TestLimits:
    def test_writes_num_frames(self, tmp_path):
        produced = tra.write_limits_frames(
            ["a", "b", "c"], tmp_path, "c" * 40, num_frames=4,
        )
        assert len(produced) == 4
        assert all(_is_png(p) for p in produced)

    def test_empty_bullets_raises(self, tmp_path):
        with pytest.raises(ValueError):
            tra.write_limits_frames([], tmp_path, "c" * 40, num_frames=3)


class TestStartIndexChaining:
    def test_timing_race_frames_respect_start_index(self, tmp_path):
        result = _sample_result(5)
        produced = tra.write_timing_race_frames(
            result=result,
            frames_dir=tmp_path,
            commit_hash="b" * 40,
            start_index=200,
        )
        assert len(produced) == 5
        names = sorted(p.name for p in produced)
        assert names[0] == "frame_00200.png"
        assert names[-1] == "frame_00204.png"


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


class TestRaceFrameIndices:
    def test_matches_write_count(self, tmp_path):
        result = _sample_result(10)
        idxs = tra.race_frame_indices(10, 3, None)
        produced = tra.write_timing_race_frames(
            result=result,
            frames_dir=tmp_path,
            commit_hash="b" * 40,
            stride=3,
        )
        assert len(produced) == len(idxs)

    def test_num_frames_mode_uniform(self):
        idxs = tra.race_frame_indices(10, num_frames=5)
        assert len(idxs) == 5
        assert idxs[0] == 0
        assert idxs[-1] == 9
        assert idxs == sorted(idxs)

    def test_num_frames_mode_oversample(self):
        idxs = tra.race_frame_indices(5, num_frames=20)
        assert len(idxs) == 20
        assert idxs[0] == 0
        assert idxs[-1] == 4

    def test_num_frames_zero_raises(self):
        with pytest.raises(ValueError):
            tra.race_frame_indices(10, num_frames=0)

    def test_num_frames_one_returns_last(self):
        assert tra.race_frame_indices(10, num_frames=1) == [9]

    def test_write_with_num_frames_mode(self, tmp_path):
        result = _sample_result(8)
        produced = tra.write_timing_race_frames(
            result=result, frames_dir=tmp_path,
            commit_hash="x" * 40, num_frames=12,
        )
        assert len(produced) == 12

    def test_telop_mismatch_raises(self, tmp_path):
        result = _sample_result(4)
        with pytest.raises(ValueError, match="telop_by_frame must have"):
            tra.write_timing_race_frames(
                result=result,
                frames_dir=tmp_path,
                commit_hash="b" * 40,
                three_pane=True,
                telop_by_frame=["a"],
            )

    def test_title_three_pane(self, tmp_path):
        produced = tra.write_titlecard_frames(
            ["a", "b"], tmp_path, "c" * 40, num_frames=2,
            three_pane=True, bottom_telop="下にテロップ。",
        )
        assert len(produced) == 2
        assert all(_is_png(p) for p in produced)
