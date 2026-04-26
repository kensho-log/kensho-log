"""VOICEVOX Engine HTTP API クライアント（標準ライブラリのみ、requests 非依存）.

起動: VOICEVOX または Engine を起動（既定 http://127.0.0.1:50021 ）

https://github.com/VOICEVOX/voicevox/blob/main/docs/VOICEX_API.md
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_BASE = "http://127.0.0.1:50021"
# 四国めたん ノーマル（0.14 系の例。/speakers で実際の id を確認すること）
DEFAULT_SPEAKER = 2


class VoicevoxError(RuntimeError):
    """API 失敗、不正レスポンス、接続失敗。"""


def _norm_base(u: str) -> str:
    u = (u or "").rstrip("/")
    if not u:
        u = DEFAULT_BASE
    if not (u.startswith("http://") or u.startswith("https://")):
        u = "http://" + u
    return u.rstrip("/")


def engine_version(base_url: str = DEFAULT_BASE) -> str | None:
    try:
        u = f"{_norm_base(base_url)}/version"
        with urllib.request.urlopen(u, timeout=5) as r:
            return r.read().decode("utf-8", errors="replace").strip()
    except (OSError, ValueError) as e:
        logger.debug("engine_version: %s", e)
        return None


def is_engine_up(base_url: str = DEFAULT_BASE) -> bool:
    return engine_version(base_url) is not None


def _get_json(u: str) -> Any:
    with urllib.request.urlopen(u, timeout=30) as r:
        return json.loads(r.read().decode("utf-8", errors="replace"))


def _http_post(
    url: str,
    body: bytes | None,
    content_type: str | None = None,
    timeout: int = 120,
) -> bytes:
    h = {"User-Agent": "kensho-log/0.1"}
    if content_type:
        h["Content-Type"] = content_type
    data = b"" if body is None else body
    req = urllib.request.Request(url, data=data, method="POST", headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", errors="replace")
        raise VoicevoxError(f"VOICEVOX HTTP {e.code} {e.reason}: {msg[:500]}") from e
    except (OSError, ValueError) as e:
        raise VoicevoxError(f"VOICEVOX 接続失敗: {e}") from e


def _audio_query(text: str, speaker: int, base_url: str) -> dict[str, Any]:
    t = (text or "").strip() or "。"
    if len(t) > 8000:
        raise ValueError("text too long for a single /audio_query (max 8000 chars)")
    q = urllib.parse.urlencode({"text": t, "speaker": str(speaker)}, encoding="utf-8")
    url = f"{_norm_base(base_url)}/audio_query?{q}"
    raw = _http_post(url, b"")
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise VoicevoxError("audio_query: JSON ではない") from e


def _synthesis(query_dict: dict[str, Any], speaker: int, base_url: str) -> bytes:
    q = urllib.parse.urlencode({"speaker": str(speaker)}, encoding="utf-8")
    url = f"{_norm_base(base_url)}/synthesis?{q}"
    body = json.dumps(query_dict, ensure_ascii=False).encode("utf-8")
    return _http_post(url, body, content_type="application/json", timeout=180)


def synthesize_wav(
    text: str,
    *,
    speaker: int = DEFAULT_SPEAKER,
    base_url: str = DEFAULT_BASE,
) -> bytes:
    """text を WAVE バイナリに合成。空文字列は「。」に置き換え。"""
    t = (text or "").strip() or "。"
    j = _audio_query(t, speaker, base_url)
    return _synthesis(j, speaker, base_url)
