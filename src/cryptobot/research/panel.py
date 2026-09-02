"""パネル（時刻 × 銘柄の行列）の構築。

研究層は全てこのパネルの上で動く。時刻は 1 時間刻みの連続した格子で、
データがない箇所は NaN。ユニバース所属は月ごとに point-in-time で決める。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from cryptobot.config import UniverseConfig
from cryptobot.data.store import DataStore
from cryptobot.data.universe import compute_daily, select_universe_from_daily

HOURS_PER_YEAR = 24 * 365


@dataclass
class Panel:
    times: np.ndarray  # datetime64[ms], 長さ T
    symbols: list[str]  # 長さ N
    close: np.ndarray  # T×N float64, NaN = データなし
    quote_volume: np.ndarray  # T×N float64
    funding: np.ndarray  # T×N float64, その時刻に適用されるファンディングレート（なければ 0）
    member: np.ndarray  # T×N bool, その時刻にユニバースに属しているか

    @property
    def n_times(self) -> int:
        return int(self.close.shape[0])

    @property
    def n_symbols(self) -> int:
        return int(self.close.shape[1])

    def slice(self, start: int, end: int) -> Panel:
        return Panel(
            self.times[start:end],
            self.symbols,
            self.close[start:end],
            self.quote_volume[start:end],
            self.funding[start:end],
            self.member[start:end],
        )

    def index_of(self, when: datetime) -> int:
        """when 以上の最初の時刻の添字。"""
        target = np.datetime64(when.astimezone(UTC).replace(tzinfo=None), "ms")
        return int(np.searchsorted(self.times, target, side="left"))


def month_starts(start: datetime, end: datetime) -> list[datetime]:
    """start を含む月の 1 日から end までの各月 1 日（UTC 00:00）。"""
    cur = datetime(start.year, start.month, 1, tzinfo=UTC)
    out: list[datetime] = []
    while cur <= end:
        out.append(cur)
        cur = datetime(cur.year + (cur.month // 12), cur.month % 12 + 1, 1, tzinfo=UTC)
    return out


def universe_schedule(
    store: DataStore, start: datetime, end: datetime, cfg: UniverseConfig
) -> dict[datetime, list[str]]:
    """月初ごとの point-in-time ユニバース。日次集計は 1 回だけ作って使い回す。"""
    daily = compute_daily(store)
    return {
        m: select_universe_from_daily(daily, m, cfg)["symbol"].to_list()
        for m in month_starts(start, end)
    }


def build_panel(
    store: DataStore,
    start: datetime,
    end: datetime,
    cfg: UniverseConfig,
    warmup_hours: int,
) -> Panel:
    """検証期間 [start, end] とその前の warmup_hours 分のデータからパネルを作る。"""
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start と end はタイムゾーン付き（UTC）にしてください")
    schedule = universe_schedule(store, start, end, cfg)
    symbols = sorted({s for names in schedule.values() for s in names})
    if not symbols:
        raise ValueError(
            "検証期間にユニバースを構成できる銘柄がありません。データを確認してください。"
        )

    data_start = start - timedelta(hours=warmup_hours)
    grid = pl.datetime_range(
        data_start, end, interval="1h", time_unit="ms", time_zone="UTC", eager=True
    ).alias("open_time")
    times = grid.to_numpy()

    kl = (
        store.scan_klines(symbols)
        .filter((pl.col("open_time") >= data_start) & (pl.col("open_time") <= end))
        .select("symbol", "open_time", "close", "quote_volume")
        .collect()
    )
    close = _pivot(kl, grid, symbols, "close")
    qv = _pivot(kl, grid, symbols, "quote_volume")

    fd = (
        store.scan_funding(symbols)
        .filter((pl.col("funding_time") >= data_start) & (pl.col("funding_time") <= end))
        .select(
            "symbol",
            pl.col("funding_time").dt.truncate("1h").alias("open_time"),
            pl.col("funding_rate"),
        )
        .group_by(["symbol", "open_time"])
        .agg(pl.col("funding_rate").sum())
        .collect()
    )
    funding = _pivot(fd, grid, symbols, "funding_rate")
    funding = np.nan_to_num(funding, nan=0.0)

    member = np.zeros(close.shape, dtype=bool)
    col = {s: j for j, s in enumerate(symbols)}
    months = sorted(schedule)
    for i, m in enumerate(months):
        m_end = months[i + 1] if i + 1 < len(months) else end + timedelta(hours=1)
        lo = int(np.searchsorted(times, np.datetime64(m.replace(tzinfo=None), "ms")))
        hi = int(np.searchsorted(times, np.datetime64(m_end.replace(tzinfo=None), "ms")))
        for s in schedule[m]:
            member[lo:hi, col[s]] = True
    # 価格がない箇所はユニバースに入れない
    member &= ~np.isnan(close)
    return Panel(times, symbols, close, qv, funding, member)


def _pivot(df: pl.DataFrame, grid: pl.Series, symbols: list[str], value: str) -> np.ndarray:
    if df.is_empty():
        return np.full((grid.len(), len(symbols)), np.nan)
    wide = (
        df.pivot(on="symbol", index="open_time", values=value)
        .join(grid.to_frame(), on="open_time", how="right")
        .sort("open_time")
    )
    cols = [
        pl.col(s).cast(pl.Float64) if s in wide.columns else pl.lit(None).cast(pl.Float64).alias(s)
        for s in symbols
    ]
    return wide.select(cols).to_numpy().astype(np.float64)
