"""narration_001 のテスト（台本・レーステロップ分割）。"""

from kensho_log.video.narration_001 import TELP_RACE_1, TELP_RACE_4, build_race_telop_lines


def test_build_race_telop_length():
    lines = build_race_telop_lines(12)
    assert len(lines) == 12
    assert lines[0] == TELP_RACE_1
    assert lines[-1] == TELP_RACE_4
