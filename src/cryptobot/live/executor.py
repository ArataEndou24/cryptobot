"""目標ポジションと現在のポジションの差分を注文にし、上限を確認して送る。"""

from __future__ import annotations

from dataclasses import dataclass, field

from cryptobot.config import LiveConfig, RiskConfig
from cryptobot.exchange.hyperliquid_meta import HLMarket
from cryptobot.exchange.hyperliquid_trader import (
    OrderRequest,
    OrderResult,
    Trader,
    round_price,
    round_size,
)


@dataclass
class ReconcileResult:
    orders: list[OrderRequest] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)


def reconcile(
    targets: dict[str, float],
    current: dict[str, float],
    mids: dict[str, float],
    markets: dict[str, HLMarket],
    equity: float,
    risk: RiskConfig,
    live: LiveConfig,
) -> ReconcileResult:
    """targets / current は銘柄 → 数量（ロング正、ショート負）。差分を IOC 指値にする。"""
    out = ReconcileResult()
    coins = sorted(set(targets) | set(current))
    gross_after = 0.0
    for coin in coins:
        tgt = targets.get(coin, 0.0)
        cur = current.get(coin, 0.0)
        mid = mids.get(coin)
        mk = markets.get(coin)
        if mid is None or mk is None:
            out.rejected.append(f"{coin}: 価格または銘柄情報がない")
            continue
        # 1 銘柄の上限（少し余裕を持たせる）
        if abs(tgt) * mid > risk.max_position_pct * equity * 1.10:
            out.rejected.append(f"{coin}: 目標 {abs(tgt) * mid:.0f} USD が 1 銘柄上限を超える")
            tgt = 0.0 if cur == 0.0 else cur
        diff = tgt - cur
        size = round_size(diff, mk.sz_decimals)
        if size <= 0:
            continue
        notional = size * mid
        if notional < live.min_order_usd:
            out.skipped.append(f"{coin}: 差分 {notional:.2f} USD が最小注文額未満")
            continue
        is_buy = diff > 0
        reduce_only = (cur > 0 and diff < 0 and tgt >= 0) or (cur < 0 and diff > 0 and tgt <= 0)
        px = mid * (1.0 + live.max_slippage_pct) if is_buy else mid * (1.0 - live.max_slippage_pct)
        out.orders.append(
            OrderRequest(coin, is_buy, size, round_price(px, mk.sz_decimals), reduce_only)
        )
        gross_after += abs(tgt) * mid
    if gross_after > risk.max_gross_leverage * equity * 1.10:
        out.rejected.append(
            f"総建玉 {gross_after:.0f} USD が上限（資産の {risk.max_gross_leverage} 倍）を超える"
            "ため全注文を見送り"
        )
        out.orders = []
    return out


def execute(trader: Trader, orders: list[OrderRequest]) -> list[OrderResult]:
    """手仕舞い（reduce_only）を先に、新規を後に送る。証拠金を先に空けるため。"""
    ordered = sorted(orders, key=lambda o: (not o.reduce_only, -o.notional))
    return [trader.place_ioc(o) for o in ordered]


def flatten_orders(
    current: dict[str, float],
    mids: dict[str, float],
    markets: dict[str, HLMarket],
    live: LiveConfig,
) -> list[OrderRequest]:
    """全ポジションを閉じる注文。"""
    out: list[OrderRequest] = []
    for coin, cur in current.items():
        mid = mids.get(coin)
        mk = markets.get(coin)
        if mid is None or mk is None or cur == 0.0:
            continue
        is_buy = cur < 0
        px = (
            mid * (1.0 + live.max_slippage_pct * 2)
            if is_buy
            else mid * (1.0 - live.max_slippage_pct * 2)
        )
        out.append(
            OrderRequest(
                coin, is_buy, round_size(cur, mk.sz_decimals), round_price(px, mk.sz_decimals), True
            )
        )
    return out
