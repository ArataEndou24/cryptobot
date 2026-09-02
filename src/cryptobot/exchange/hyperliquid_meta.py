"""Hyperliquid の上場銘柄情報（読み取りのみ。鍵は不要）。

研究データは Binance の銘柄名、取引は Hyperliquid の銘柄名なので、対応表が要る。
Binance の "1000PEPEUSDT"（1000 枚単位）は Hyperliquid では "kPEPE"（k = 1000）に相当する。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

INFO_URL = {
    "mainnet": "https://api.hyperliquid.xyz/info",
    "testnet": "https://api.hyperliquid-testnet.xyz/info",
}
CACHE_NAME = "hyperliquid_symbols.json"


@dataclass(frozen=True)
class HLMarket:
    name: str
    max_leverage: int
    sz_decimals: int


def fetch_markets(network: str = "mainnet", timeout: float = 20.0) -> list[HLMarket]:
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(INFO_URL[network], json={"type": "meta"})
        resp.raise_for_status()
        universe = resp.json()["universe"]
    return [
        HLMarket(u["name"], int(u.get("maxLeverage", 1)), int(u.get("szDecimals", 0)))
        for u in universe
        if not u.get("isDelisted")
    ]


def cache_path(data_root: Path) -> Path:
    return data_root / CACHE_NAME


def save_markets(markets: list[HLMarket], data_root: Path, network: str) -> Path:
    path = cache_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "network": network,
        "fetched_at": datetime.now(UTC).isoformat(),
        "markets": [m.__dict__ for m in markets],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def load_markets(data_root: Path) -> tuple[list[HLMarket], str | None]:
    """キャッシュを読む。なければ空リストと None を返す。"""
    path = cache_path(data_root)
    if not path.exists():
        return [], None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [HLMarket(**m) for m in payload["markets"]], str(payload.get("fetched_at"))


def binance_to_hl(symbol: str, quote: str = "USDT") -> list[str]:
    """Binance 銘柄名から Hyperliquid 銘柄名の候補を返す（存在確認は呼び手が行う）。"""
    base = symbol.removesuffix(quote)
    candidates: list[str] = []
    for prefix in ("1000000", "1000", "1M"):
        if base.startswith(prefix) and len(base) > len(prefix):
            rest = base[len(prefix) :]
            candidates += [f"k{rest}", rest]
    candidates.append(base)
    return list(dict.fromkeys(candidates))


def map_symbols(
    binance_symbols: list[str], markets: list[HLMarket], quote: str = "USDT"
) -> dict[str, str]:
    """Binance 銘柄名 → Hyperliquid 銘柄名。対応がないものは含めない。"""
    names = {m.name for m in markets}
    out: dict[str, str] = {}
    for b in binance_symbols:
        for cand in binance_to_hl(b, quote):
            if cand in names:
                out[b] = cand
                break
    return out
