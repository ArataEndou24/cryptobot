"""Hyperliquid への注文（取引専用の API ウォレットで署名）。

安全設計:
- 鍵は環境変数 / .env からだけ読む。ログにも例外にも鍵を出さない。
- この鍵は「注文はできるが出金はできない」API ウォレットのもの。メインウォレットの鍵は使わない。
- 価格は取引所の丸め規則（有効数字 5 桁、小数は 6 - szDecimals 桁まで）に合わせる。
- 注文は IOC（即時約定しなければ取り消し）の指値で、滑りの上限を必ず付ける。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol

from cryptobot.exchange.hyperliquid_meta import HLMarket


@dataclass
class OrderRequest:
    coin: str
    is_buy: bool
    size: float
    limit_px: float
    reduce_only: bool = False

    @property
    def notional(self) -> float:
        return self.size * self.limit_px


@dataclass
class OrderResult:
    request: OrderRequest
    ok: bool
    filled_size: float = 0.0
    avg_px: float | None = None
    message: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


class Trader(Protocol):
    """執行層が依存する最小の取引所インターフェース。テストでは偽物に差し替える。"""

    def markets(self) -> dict[str, HLMarket]: ...
    def mids(self) -> dict[str, float]: ...
    def equity(self) -> float: ...
    def positions(self) -> dict[str, float]: ...
    def place_ioc(self, req: OrderRequest) -> OrderResult: ...
    def cancel_all(self) -> int: ...


def round_price(px: float, sz_decimals: int) -> float:
    """Hyperliquid の価格丸め: 有効数字 5 桁、かつ小数は (6 - szDecimals) 桁まで。"""
    if px <= 0:
        raise ValueError("価格は正の値である必要があります")
    max_decimals = max(0, 6 - sz_decimals)
    px = float(f"{px:.5g}")
    return round(px, max_decimals)


def round_size(size: float, sz_decimals: int) -> float:
    step = 10.0 ** (-sz_decimals)
    return math.floor(abs(size) / step + 1e-9) * step


class HyperliquidTrader:
    """SDK を使う本物の実装。"""

    def __init__(self, network: str, agent_private_key: str, main_address: str) -> None:
        from eth_account import Account
        from hyperliquid.exchange import Exchange
        from hyperliquid.info import Info
        from hyperliquid.utils import constants

        base = constants.MAINNET_API_URL if network == "mainnet" else constants.TESTNET_API_URL
        wallet = Account.from_key(agent_private_key)
        self.address = main_address
        self.network = network
        self._info = Info(base, skip_ws=True)
        self._exchange = Exchange(wallet, base, account_address=main_address)
        self._markets: dict[str, HLMarket] | None = None

    def markets(self) -> dict[str, HLMarket]:
        if self._markets is None:
            meta = self._info.meta()
            self._markets = {
                u["name"]: HLMarket(u["name"], int(u.get("maxLeverage", 1)), int(u["szDecimals"]))
                for u in meta["universe"]
                if not u.get("isDelisted")
            }
        return self._markets

    def mids(self) -> dict[str, float]:
        raw = self._info.all_mids()
        return {k: float(v) for k, v in raw.items() if not k.startswith("@")}

    def user_state(self) -> dict[str, Any]:
        result: dict[str, Any] = self._info.user_state(self.address)
        return result

    def equity(self) -> float:
        return float((self.user_state().get("marginSummary") or {}).get("accountValue") or 0.0)

    def positions(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for ap in self.user_state().get("assetPositions") or []:
            pos = ap.get("position") or {}
            szi = float(pos.get("szi") or 0.0)
            if szi != 0.0:
                out[str(pos["coin"])] = szi
        return out

    def place_ioc(self, req: OrderRequest) -> OrderResult:
        try:
            raw = self._exchange.order(
                req.coin,
                req.is_buy,
                req.size,
                req.limit_px,
                {"limit": {"tif": "Ioc"}},
                reduce_only=req.reduce_only,
            )
        except Exception as e:  # SDK の例外はそのまま結果に載せる（鍵は含まれない）
            return OrderResult(req, False, message=f"{type(e).__name__}: {e}")
        return parse_order_response(req, raw)

    def cancel_all(self) -> int:
        n = 0
        for o in self._info.open_orders(self.address):
            try:
                self._exchange.cancel(o["coin"], int(o["oid"]))
                n += 1
            except Exception:
                pass
        return n


def parse_order_response(req: OrderRequest, raw: Any) -> OrderResult:
    """SDK の応答を共通形式にする。応答例:
    {"status":"ok","response":{"type":"order","data":{"statuses":[{"filled":{"totalSz":"0.1","avgPx":"100.0","oid":1}}]}}}
    {"status":"ok","response":{"type":"order","data":{"statuses":[{"error":"..."}]}}}
    """
    try:
        if raw.get("status") != "ok":
            return OrderResult(req, False, message=str(raw), raw=raw)
        statuses = raw["response"]["data"]["statuses"]
        st = statuses[0] if statuses else {}
        if "filled" in st:
            f = st["filled"]
            return OrderResult(
                req, True, float(f.get("totalSz", 0.0)), float(f.get("avgPx", 0.0)), "約定", raw
            )
        if "resting" in st:
            return OrderResult(req, True, 0.0, None, "板に残った（IOC では起きないはず）", raw)
        if "error" in st:
            return OrderResult(req, False, message=str(st["error"]), raw=raw)
        return OrderResult(req, False, message=f"不明な応答: {st}", raw=raw)
    except (KeyError, TypeError, AttributeError, IndexError) as e:
        return OrderResult(req, False, message=f"応答の解釈に失敗: {e}: {raw}")
