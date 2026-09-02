"""環境診断。運用者が `make doctor` で実行し、赤い項目があれば報告する。"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx

from cryptobot.config import ConfigError, Settings, load_settings, resolve_config_path

HL_API = {
    "mainnet": "https://api.hyperliquid.xyz/info",
    "testnet": "https://api.hyperliquid-testnet.xyz/info",
}
BINANCE_PROBE = (
    "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2024-01.zip"
)
MIN_FREE_GB = 5.0


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def run_checks(config_path: Path | None = None) -> list[Check]:
    checks: list[Check] = []

    v = sys.version_info
    checks.append(
        Check(
            "Python のバージョン",
            (v.major, v.minor) >= (3, 12),
            f"{v.major}.{v.minor}.{v.micro}（3.12 以上が必要）",
        )
    )

    settings: Settings | None = None
    resolved, fallback = resolve_config_path(config_path)
    try:
        settings = load_settings(config_path)
        note = "（例ファイルを使用中。`make setup` で自分用の設定を作成）" if fallback else ""
        checks.append(Check("設定ファイル", True, f"{resolved} {note}".strip()))
    except ConfigError as e:
        checks.append(Check("設定ファイル", False, str(e)))

    if settings is not None:
        root = settings.data.root
        try:
            root.mkdir(parents=True, exist_ok=True)
            probe = root / ".write_test"
            probe.write_text("ok")
            probe.unlink()
            checks.append(Check("データ保存先への書き込み", True, str(root.resolve())))
        except OSError as e:
            checks.append(Check("データ保存先への書き込み", False, f"{root}: {e}"))

        usage = shutil.disk_usage(root if root.exists() else Path.cwd())
        free_gb = usage.free / 1e9
        checks.append(
            Check(
                "ディスクの空き容量",
                free_gb >= MIN_FREE_GB,
                f"{free_gb:.1f} GB（{MIN_FREE_GB:.0f} GB 以上を推奨）",
            )
        )

    with httpx.Client(timeout=15, follow_redirects=True) as client:
        try:
            r = client.head(BINANCE_PROBE)
            checks.append(
                Check(
                    "Binance 過去データ配布サーバー", r.status_code == 200, f"HTTP {r.status_code}"
                )
            )
        except httpx.HTTPError as e:
            checks.append(Check("Binance 過去データ配布サーバー", False, str(e)))

        network = settings.exchange.network if settings else "testnet"
        try:
            r = client.post(HL_API[network], json={"type": "meta"})
            n = len(r.json().get("universe", [])) if r.status_code == 200 else 0
            checks.append(
                Check(
                    f"Hyperliquid API（{network}）",
                    r.status_code == 200 and n > 0,
                    f"HTTP {r.status_code}、銘柄数 {n}",
                )
            )
        except (httpx.HTTPError, ValueError) as e:
            checks.append(Check(f"Hyperliquid API（{network}）", False, str(e)))

    env = Path(".env")
    checks.append(
        Check(
            "秘密情報ファイル .env",
            True,
            "あり（内容は表示しません）"
            if env.exists()
            else "なし（取引を始める段階で作成します）",
        )
    )
    return checks


def format_report(checks: list[Check]) -> tuple[str, bool]:
    lines = ["環境診断の結果", "=" * 40]
    all_ok = True
    for c in checks:
        mark = "OK " if c.ok else "NG "
        all_ok &= c.ok
        lines.append(f"[{mark}] {c.name}: {c.detail}")
    lines.append("=" * 40)
    lines.append(
        "全て正常です。"
        if all_ok
        else "NG の項目があります。この出力をそのまま貼り付けて報告してください。"
    )
    return "\n".join(lines), all_ok
