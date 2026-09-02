"""Hyperliquid の公開情報 API（読み取りのみ。鍵は不要）。

- 1 時間足（最大 5000 本/回。2200 本を 1 回で取れる）
- 銘柄ごとの現在値、ファンディング、建玉、日次出来高
- 口座状態（アドレスを指定。公開情報なので鍵は不要）
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import polars as pl

from cryptobot.exchange.hyperliquid_meta import INFO_URL, HLMarket


@dataclass(frozen=True)
class AssetContext:
    name: str
    mark_px: float
    mid_px: float
    funding_8h: float
    open_interest: float
    day_notional_volume: float
    max_leverage: int
    sz_decimals: int


class HyperliquidInfo:
    def __init__(self, network: str = "mainnet", timeout: float = 30.0) -> None:
        self.url = INFO_URL[network]
        self.client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> HyperliquidInfo:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _post(self, payload: dict[str, Any]) -> Any:
        resp = self.client.post(self.url, json=payload)
        resp.raise_for_status()
        return resp.json()

    def asset_contexts(self) -> list[AssetContext]:
        meta, ctxs = self._post({"type": "metaAndAssetCtxs"})
        out: list[AssetContext] = []
        for u, c in zip(meta["universe"], ctxs, strict=True):
            if u.get("isDelisted"):
                continue
            out.append(
                AssetContext(
                    name=u["name"],
                    mark_px=float(c["markPx"]),
                    mid_px=float(c.get("midPx") or c["markPx"]),
                    funding_8h=float(c.get("funding") or 0.0),
                    open_interest=float(c.get("openInterest") or 0.0),
                    day_notional_volume=float(c.get("dayNtlVlm") or 0.0),
                    max_leverage=int(u.get("maxLeverage", 1)),
                    sz_decimals=int(u.get("szDecimals", 0)),
                )
            )
        return out

    def markets(self) -> list[HLMarket]:
        return [HLMarket(c.name, c.max_leverage, c.sz_decimals) for c in self.asset_contexts()]

    def candles(self, coin: str, hours: int, end: datetime | None = None) -> pl.DataFrame:
        """直近 hours 本の 1 時間足。列: open_time, open, high, low, close, volume, trades."""
        end = end or datetime.now(UTC)
        start = end - timedelta(hours=hours)
        raw = self._post(
            {
                "type": "candleSnapshot",
                "req": {
                    "coin": coin,
                    "interval": "1h",
                    "startTime": int(start.timestamp() * 1000),
                    "endTime": int(end.timestamp() * 1000),
                },
            }
        )
        return parse_candles(raw)

    def user_state(self, address: str) -> dict[str, Any]:
        result: dict[str, Any] = self._post({"type": "clearinghouseState", "user": address})
        return result


def parse_candles(raw: list[dict[str, Any]]) -> pl.DataFrame:
    if not raw:
        return pl.DataFrame(
            schema={
                "open_time": pl.Datetime("ms", "UTC"),
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
                "trades": pl.Int64,
            }
        )
    df = pl.DataFrame(
        {
            "open_time": [int(r["t"]) for r in raw],
            "open": [float(r["o"]) for r in raw],
            "high": [float(r["h"]) for r in raw],
            "low": [float(r["l"]) for r in raw],
            "close": [float(r["c"]) for r in raw],
            "volume": [float(r["v"]) for r in raw],
            "trades": [int(r["n"]) for r in raw],
        }
    )
    return df.with_columns(
        pl.from_epoch("open_time", time_unit="ms")
        .dt.cast_time_unit("ms")
        .dt.replace_time_zone("UTC")
    ).sort("open_time")


def account_equity(state: dict[str, Any]) -> float:
    """口座の証拠金残高（USDC 建て）。"""
    summary = state.get("marginSummary") or {}
    return float(summary.get("accountValue") or 0.0)


def open_positions(state: dict[str, Any]) -> dict[str, float]:
    """銘柄 → 保有数量（ロング正、ショート負）。"""
    out: dict[str, float] = {}
    for ap in state.get("assetPositions") or []:
        pos = ap.get("position") or {}
        szi = float(pos.get("szi") or 0.0)
        if szi != 0.0:
            out[str(pos.get("coin"))] = szi
    return out
