"""仮説 001: 台本（テロップ＋TTS 一括文字列）.

YMYL: 投資推奨・最適化の語は使わない。条件と結果の記述のみ。
"""

from __future__ import annotations

# 一括 TTS 用。句点で区切る（VOICEVOX が自然に区切る）
TTS_FULL_DEFAULT = (
    "検証ログ。仮説ゼロゼロいち、SアンドPファイブハンドレッド、"
    "年間同額を完璧月と毎月積立と最悪月の三案で、"
    "配当再投資込みのリターンを比較した。 "
    "Aは各年最安月に一括。Bは毎月定額。Cは各年最高月の一括。 "
    "レース中は、ドットコム、リーマン、コロナの前後にマーカーがある。 "
    "最後に最終円額の差、検証限界、コミットで再現性を示す。 "
    "本動画は助言を目的とせず、条件と試算結果の記録にすぎない。"
)

TELP_COLD = (
    "検証ログ。仮説001。S&P 500 配当再投資込のリターンを、三案で月次比較する。"
)
TELP_CONDITIONS = (
    "A:各年最安月一括  B:毎月定額  C:各年最高月一括。年間投じる元金は同額。"
)
TELP_RACE_1 = (
    "レース。横は年月、縦は評価円。薄い点線は累積投じた元本。"
)
TELP_RACE_2 = (
    "二〇〇〇年前後はドットコム、〇八年はGFC。高値一括のCの負傷深さに注目。"
)
TELP_RACE_3 = (
    "二〇年はコロナ急落と反発。Bは毎月買い、高値一括Cとの差の材料になる。"
)
TELP_RACE_4 = (
    "最終月へ。AとB、Cの順位感は、暴落前後の差し引きの結果で見える。"
)
TELP_FINALE = (
    "最終評価。元本同額。A減Bの差は画面中央。外れ値や税は含めていない。"
)
TELP_LIMITS = (
    "限界:後知恵、税手数料為替、将来重複。過去は将来の保証でない。Voice:VOICEVOX/四国めたん"
)


def build_race_telop_lines(num_frames: int) -> list[str]:
    """レース フレーム数分のテロップ。四分割で行を入れ替え（台本は固定文字列）。"""
    n = int(num_frames)
    if n < 1:
        raise ValueError("num_frames must be >= 1")
    segs = [TELP_RACE_1, TELP_RACE_2, TELP_RACE_3, TELP_RACE_4]
    out: list[str] = []
    for i in range(n):
        q = int((i * 4) // n)  # 0..3
        if q > 3:
            q = 3
        out.append(segs[q])
    return out
