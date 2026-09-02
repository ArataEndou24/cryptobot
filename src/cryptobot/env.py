"""秘密情報（.env）の読み込み。依存ライブラリを増やさないための最小実装。"""

from __future__ import annotations

import os
from pathlib import Path


def load_env(path: Path = Path(".env")) -> dict[str, str]:
    """.env を読み、未設定の環境変数だけを補う。値は返すが表示してはいけない。"""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key:
            values[key] = value
            os.environ.setdefault(key, value)
    return values


def secret(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None
