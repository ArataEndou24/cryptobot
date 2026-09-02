"""運用状態の永続化と安全装置（キルスイッチ）。

状態は JSON 1 ファイル（data/live/state.json）。運用者が読めるように日本語のキー説明を併記する。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from cryptobot.config import RiskConfig


@dataclass
class LiveState:
    peak_equity: float = 0.0
    day: str = ""  # UTC の日付 YYYY-MM-DD
    day_start_equity: float = 0.0
    last_equity: float = 0.0
    last_run_at: str = ""
    halted: bool = False
    halt_reason: str = ""
    consecutive_errors: int = 0
    history: list[dict[str, float | str]] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> LiveState:
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        data.pop("_説明", None)
        return cls(**data)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "_説明": (
                "運用状態。halted が true なら新規注文を出さない。"
                "手動で再開するには `cryptobot live resume`。"
            ),
            **asdict(self),
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)


@dataclass
class SafetyVerdict:
    allow_new_orders: bool
    flatten: bool
    halt: bool
    reason: str


def update_and_check(
    state: LiveState, equity: float, risk: RiskConfig, now: datetime | None = None
) -> SafetyVerdict:
    """資産を記録し、日次損失と最大ドローダウンの上限を確認する。"""
    now = now or datetime.now(UTC)
    today = now.strftime("%Y-%m-%d")
    if state.day != today:
        state.day = today
        state.day_start_equity = equity
    if state.peak_equity <= 0:
        state.peak_equity = equity
    state.peak_equity = max(state.peak_equity, equity)
    state.last_equity = equity
    state.last_run_at = now.isoformat()
    state.history.append({"at": now.isoformat(), "equity": equity})
    state.history = state.history[-2000:]

    if state.halted:
        return SafetyVerdict(False, False, True, f"停止中: {state.halt_reason}")
    drawdown = 1.0 - equity / state.peak_equity if state.peak_equity > 0 else 0.0
    if drawdown >= risk.max_drawdown_pct:
        state.halted = True
        state.halt_reason = (
            f"最大ドローダウン {drawdown:.1%} が上限 {risk.max_drawdown_pct:.0%} に達した"
        )
        return SafetyVerdict(False, True, True, state.halt_reason)
    daily = 1.0 - equity / state.day_start_equity if state.day_start_equity > 0 else 0.0
    if daily >= risk.max_daily_loss_pct:
        reason = (
            f"本日の損失 {daily:.1%} が上限 {risk.max_daily_loss_pct:.0%} に達した"
            "（翌日まで新規注文なし）"
        )
        return SafetyVerdict(False, True, False, reason)
    return SafetyVerdict(True, False, False, "正常")
