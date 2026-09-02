"""1 回のリバランス周期を実行する。

順序:
1. 口座残高を取り、安全装置（日次損失、最大ドローダウン）を確認する。
2. 目標ポジションを計算する（planner）。
3. 現在のポジションとの差分を注文にし、上限を確認する（executor）。
4. 送信し、結果を記録し、日本語の要約を返す。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from cryptobot.config import Settings
from cryptobot.data.store import DataStore
from cryptobot.exchange.hyperliquid_info import HyperliquidInfo
from cryptobot.exchange.hyperliquid_trader import OrderRequest, OrderResult, Trader
from cryptobot.live.executor import execute, flatten_orders, reconcile
from cryptobot.live.planner import Plan, build_plan
from cryptobot.live.state import LiveState, update_and_check


@dataclass
class CycleReport:
    at: datetime
    equity: float
    verdict: str
    plan: Plan | None
    orders: list[OrderRequest]
    results: list[OrderResult]
    skipped: list[str]
    rejected: list[str]
    executed: bool
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"[{self.at:%Y-%m-%d %H:%M} UTC] 資産 {self.equity:,.2f} USD、安全装置: {self.verdict}"
        ]
        if self.plan is not None:
            gross = self.plan.gross_notional
            lev = gross / max(self.plan.equity, 1e-9)
            n_pos = len(self.plan.positions)
            lines.append(f"  目標: {n_pos} 銘柄、総建玉 {gross:,.0f} USD（資産の {lev:.2f} 倍）")
        mode = "送信" if self.executed else "乾式運転（送信せず）"
        lines.append(f"  注文 {len(self.orders)} 件（{mode}）")
        for o in self.orders:
            side = "買" if o.is_buy else "売"
            ro = "（手仕舞い）" if o.reduce_only else ""
            head = f"    {o.coin:<8} {side} {o.size:g} @ {o.limit_px:g}"
            lines.append(f"{head}（約 {o.notional:,.0f} USD）{ro}")
        for r in self.results:
            mark = "OK" if r.ok else "NG"
            fill = ""
            if r.ok and r.avg_px:
                fill = f" 約定 {r.filled_size:g} @ {r.avg_px:g}"
            lines.append(f"    [{mark}] {r.request.coin}: {r.message}{fill}")
        for s in self.skipped:
            lines.append(f"  見送り: {s}")
        for s in self.rejected:
            lines.append(f"  拒否: {s}")
        lines.extend(f"  注意: {n}" for n in self.notes)
        return "\n".join(lines)


def run_cycle(
    settings: Settings,
    store: DataStore,
    trader: Trader,
    market_info: HyperliquidInfo,
    execute_orders: bool,
    now: datetime | None = None,
) -> CycleReport:
    now = now or datetime.now(UTC)
    state_path = settings.live.state_dir / "state.json"
    state = LiveState.load(state_path)
    equity = trader.equity()
    verdict = update_and_check(state, equity, settings.risk, now)
    report = CycleReport(now, equity, verdict.reason, None, [], [], [], [], execute_orders)
    try:
        current = trader.positions()
        mids = trader.mids()
        markets = trader.markets()
        if verdict.flatten:
            report.orders = flatten_orders(current, mids, markets, settings.live)
            report.notes.append("安全装置が作動したため全ポジションを閉じます")
        elif verdict.allow_new_orders:
            plan = build_plan(settings, store, market_info, equity)
            report.plan = plan
            targets = {p.coin: p.size for p in plan.positions}
            rec = reconcile(targets, current, mids, markets, equity, settings.risk, settings.live)
            report.orders, report.skipped, report.rejected = rec.orders, rec.skipped, rec.rejected
            report.skipped.extend(plan.skipped)
        if execute_orders and report.orders:
            report.results = execute(trader, report.orders)
            failures = [r for r in report.results if not r.ok]
            state.consecutive_errors = len(failures) if failures else 0
        _append_log(settings.live.state_dir / "cycles.jsonl", report)
    except Exception as e:  # 1 周期の失敗で止めず、記録して次の周期に回す
        state.consecutive_errors += 1
        report.notes.append(f"エラー: {type(e).__name__}: {e}")
        if state.consecutive_errors >= 3:
            state.halted = True
            state.halt_reason = f"連続 {state.consecutive_errors} 回のエラー: {type(e).__name__}"
            report.notes.append(
                "連続エラーのため停止しました。`cryptobot live resume` で再開できます"
            )
    finally:
        state.save(state_path)
    return report


def _append_log(path: Path, report: CycleReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "at": report.at.isoformat(),
        "equity": report.equity,
        "verdict": report.verdict,
        "executed": report.executed,
        "orders": [
            {
                "coin": o.coin,
                "buy": o.is_buy,
                "size": o.size,
                "px": o.limit_px,
                "reduce_only": o.reduce_only,
            }
            for o in report.orders
        ],
        "results": [
            {
                "coin": r.request.coin,
                "ok": r.ok,
                "filled": r.filled_size,
                "avg_px": r.avg_px,
                "msg": r.message,
            }
            for r in report.results
        ],
        "skipped": report.skipped,
        "rejected": report.rejected,
        "notes": report.notes,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
