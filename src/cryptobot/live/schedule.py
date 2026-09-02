"""リバランス時刻の計算。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta


def next_run_time(now: datetime, rebalance_hours: int, delay_minutes: int) -> datetime:
    """now 以降で最初の「UTC の rebalance_hours の倍数の時刻 + delay_minutes」。"""
    now = now.astimezone(UTC)
    base = now.replace(minute=0, second=0, microsecond=0)
    base = base - timedelta(hours=base.hour % rebalance_hours)
    candidate = base + timedelta(minutes=delay_minutes)
    while candidate <= now:
        candidate += timedelta(hours=rebalance_hours)
    return candidate
