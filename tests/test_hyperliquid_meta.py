from __future__ import annotations

from pathlib import Path

from cryptobot.exchange.hyperliquid_meta import (
    HLMarket,
    binance_to_hl,
    load_markets,
    map_symbols,
    save_markets,
)


def test_name_mapping() -> None:
    assert binance_to_hl("BTCUSDT") == ["BTC"]
    assert binance_to_hl("1000PEPEUSDT") == ["kPEPE", "PEPE", "1000PEPE"]
    assert binance_to_hl("1000000MOGUSDT")[0] == "kMOG"
    markets = [HLMarket("BTC", 40, 5), HLMarket("kPEPE", 10, 0), HLMarket("MOG", 5, 0)]
    m = map_symbols(["BTCUSDT", "1000PEPEUSDT", "1000000MOGUSDT", "TUTUSDT"], markets)
    assert m == {"BTCUSDT": "BTC", "1000PEPEUSDT": "kPEPE", "1000000MOGUSDT": "MOG"}


def test_cache_roundtrip(tmp_path: Path) -> None:
    markets = [HLMarket("BTC", 40, 5), HLMarket("ETH", 25, 4)]
    save_markets(markets, tmp_path, "mainnet")
    loaded, fetched_at = load_markets(tmp_path)
    assert loaded == markets
    assert fetched_at is not None
    assert load_markets(tmp_path / "nope") == ([], None)
