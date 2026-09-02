from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from cryptobot.config import LiveConfig, RiskConfig
from cryptobot.exchange.hyperliquid_meta import HLMarket
from cryptobot.exchange.hyperliquid_trader import (
    OrderRequest,
    OrderResult,
    parse_order_response,
    round_price,
    round_size,
)
from cryptobot.live.executor import execute, flatten_orders, reconcile
from cryptobot.live.schedule import next_run_time
from cryptobot.live.state import LiveState, update_and_check

MARKETS = {
    "BTC": HLMarket("BTC", 40, 5),
    "DOGE": HLMarket("DOGE", 10, 0),
    "ETH": HLMarket("ETH", 25, 4),
}
MIDS = {"BTC": 77012.3456, "DOGE": 0.081234, "ETH": 2387.65}


def test_round_price_follows_exchange_rules() -> None:
    assert round_price(77012.3456, 5) == 77012.0  # 有効数字 5 桁
    assert round_price(0.081234, 0) == 0.081234  # 小数 6 桁まで
    assert round_price(0.0812345678, 0) == 0.081235
    assert round_price(2387.65, 4) == 2387.6 or round_price(2387.65, 4) == 2387.7
    assert round_size(0.123456789, 5) == pytest.approx(0.12345)
    assert round_size(-3.7, 0) == 3.0


def test_parse_order_response() -> None:
    req = OrderRequest("BTC", True, 0.001, 77000.0)
    ok = {
        "status": "ok",
        "response": {
            "type": "order",
            "data": {"statuses": [{"filled": {"totalSz": "0.001", "avgPx": "76990.0", "oid": 1}}]},
        },
    }
    r = parse_order_response(req, ok)
    assert r.ok and r.filled_size == 0.001 and r.avg_px == 76990.0
    err = {
        "status": "ok",
        "response": {"type": "order", "data": {"statuses": [{"error": "Insufficient margin"}]}},
    }
    assert not parse_order_response(req, err).ok
    assert not parse_order_response(req, {"status": "err", "response": "bad"}).ok
    assert not parse_order_response(req, None).ok


def _live() -> LiveConfig:
    return LiveConfig(min_order_usd=10.0, max_slippage_pct=0.003)


def test_reconcile_builds_diff_orders() -> None:
    risk = RiskConfig(max_position_pct=0.15, max_gross_leverage=2.0)
    targets = {"BTC": 0.002, "DOGE": -1000.0}
    current = {"BTC": 0.001, "ETH": 0.05}
    rec = reconcile(targets, current, MIDS, MARKETS, equity=1300.0, risk=risk, live=_live())
    by_coin = {o.coin: o for o in rec.orders}
    assert by_coin["BTC"].is_buy and by_coin["BTC"].size == pytest.approx(0.001)
    assert by_coin["BTC"].limit_px > MIDS["BTC"]  # 買いは現在値より少し上に上限
    assert not by_coin["DOGE"].is_buy and by_coin["DOGE"].size == 1000.0
    assert not by_coin["ETH"].is_buy and by_coin["ETH"].reduce_only  # 目標 0 → 手仕舞い
    assert by_coin["ETH"].limit_px < MIDS["ETH"]
    assert rec.rejected == []


def test_reconcile_skips_tiny_and_rejects_oversized() -> None:
    risk = RiskConfig(max_position_pct=0.15, max_gross_leverage=2.0)
    rec = reconcile({"DOGE": 50.0}, {}, MIDS, MARKETS, 1300.0, risk, _live())
    assert rec.orders == [] and any("最小注文額" in s for s in rec.skipped)
    rec = reconcile({"BTC": 0.01}, {}, MIDS, MARKETS, 1300.0, risk, _live())  # 770 USD > 15%
    assert rec.orders == [] and any("1 銘柄上限" in s for s in rec.rejected)
    tight = RiskConfig(max_position_pct=0.15, max_gross_leverage=0.3)  # 総建玉上限 330 USD
    targets = {"BTC": 0.002, "ETH": 0.06, "DOGE": -1900.0}
    rec = reconcile(targets, {}, MIDS, MARKETS, 1000.0, tight, _live())
    assert rec.orders == [] and any("総建玉" in s for s in rec.rejected)


def test_flatten_orders_are_reduce_only_and_opposite() -> None:
    orders = flatten_orders({"BTC": 0.002, "DOGE": -500.0, "ETH": 0.0}, MIDS, MARKETS, _live())
    assert {o.coin for o in orders} == {"BTC", "DOGE"}
    for o in orders:
        assert o.reduce_only
    btc = next(o for o in orders if o.coin == "BTC")
    assert not btc.is_buy and btc.size == 0.002


class FakeTrader:
    def __init__(self) -> None:
        self.sent: list[OrderRequest] = []
        self.fail_coins: set[str] = set()

    def markets(self) -> dict[str, HLMarket]:
        return MARKETS

    def mids(self) -> dict[str, float]:
        return MIDS

    def equity(self) -> float:
        return 1300.0

    def positions(self) -> dict[str, float]:
        return {}

    def place_ioc(self, req: OrderRequest) -> OrderResult:
        self.sent.append(req)
        if req.coin in self.fail_coins:
            return OrderResult(req, False, message="rejected")
        return OrderResult(req, True, req.size, req.limit_px, "約定")

    def cancel_all(self) -> int:
        return 0


def test_execute_sends_reduce_only_first() -> None:
    t = FakeTrader()
    orders = [
        OrderRequest("BTC", True, 0.002, 77000.0, False),
        OrderRequest("ETH", False, 0.05, 2380.0, True),
        OrderRequest("DOGE", False, 1000.0, 0.081, False),
    ]
    results = execute(t, orders)
    assert t.sent[0].coin == "ETH" and t.sent[0].reduce_only
    assert all(r.ok for r in results)


def test_safety_daily_loss_and_drawdown(tmp_path: Path) -> None:
    risk = RiskConfig(max_daily_loss_pct=0.08, max_drawdown_pct=0.40)
    st = LiveState()
    t0 = datetime(2026, 9, 2, 0, 5, tzinfo=UTC)
    v = update_and_check(st, 1000.0, risk, t0)
    assert v.allow_new_orders and not v.flatten
    v = update_and_check(st, 915.0, risk, t0 + timedelta(hours=4))  # 本日 -8.5%
    assert not v.allow_new_orders and v.flatten and not v.halt and not st.halted
    v = update_and_check(st, 950.0, risk, t0 + timedelta(days=1))  # 翌日: 日次はリセット
    assert v.allow_new_orders
    v = update_and_check(st, 590.0, risk, t0 + timedelta(days=2))  # 最高値 1000 から -41%
    assert v.halt and v.flatten and st.halted
    st.save(tmp_path / "state.json")
    loaded = LiveState.load(tmp_path / "state.json")
    assert loaded.halted and loaded.peak_equity == 1000.0
    v = update_and_check(loaded, 900.0, risk, t0 + timedelta(days=3))
    assert not v.allow_new_orders and v.halt


def test_next_run_time() -> None:
    now = datetime(2026, 9, 2, 3, 59, tzinfo=UTC)
    assert next_run_time(now, 4, 3) == datetime(2026, 9, 2, 4, 3, tzinfo=UTC)
    now = datetime(2026, 9, 2, 4, 3, 30, tzinfo=UTC)
    assert next_run_time(now, 4, 3) == datetime(2026, 9, 2, 8, 3, tzinfo=UTC)
    now = datetime(2026, 9, 2, 23, 30, tzinfo=UTC)
    assert next_run_time(now, 24, 3) == datetime(2026, 9, 3, 0, 3, tzinfo=UTC)
