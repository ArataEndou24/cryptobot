"""今この時点の目標ポジションを作る（乾式運転）。

流れ:
1. ユニバース: Binance 日次集計から point-in-time で上位 N を選び、Hyperliquid 上場銘柄に絞る。
   （Binance の配布データは 1〜2 日遅れるが、月単位の選定には十分）
2. シグナル: Hyperliquid の 1 時間足を直近 2200 本取り、研究層と同じ関数で目標ウェイトを計算する。
3. サイズ: 口座残高 × ウェイト ÷ 現在値を、取引所の数量刻みに丸める。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import polars as pl

from cryptobot.config import Settings
from cryptobot.data.store import DataStore
from cryptobot.data.universe import compute_daily, select_universe_from_daily
from cryptobot.exchange.hyperliquid_info import AssetContext, HyperliquidInfo
from cryptobot.exchange.hyperliquid_meta import HLMarket, map_symbols
from cryptobot.research.panel import Panel
from cryptobot.strategy.momentum import momentum_scores, target_weights

MIN_ORDER_NOTIONAL_USD = 10.0


@dataclass
class PlannedPosition:
    binance_symbol: str
    coin: str
    score: float
    weight: float
    notional_usd: float
    size: float
    mark_px: float
    funding_8h: float


@dataclass
class Plan:
    as_of: datetime
    equity: float
    universe: list[str]
    positions: list[PlannedPosition]
    skipped: list[str]

    @property
    def gross_notional(self) -> float:
        return sum(abs(p.notional_usd) for p in self.positions)


def build_plan(
    settings: Settings,
    store: DataStore,
    info: HyperliquidInfo,
    equity: float,
    hours: int = 2200,
) -> Plan:
    as_of = datetime.now(UTC)
    ctxs = {c.name: c for c in info.asset_contexts()}
    markets = [_market_of(c) for c in ctxs.values()]
    daily = compute_daily(store)
    uni = select_universe_from_daily(daily, as_of, settings.universe)
    binance_symbols = uni["symbol"].to_list()
    mapping = map_symbols(binance_symbols, markets, settings.universe.quote)
    skipped = [s for s in binance_symbols if s not in mapping]
    coins = [mapping[s] for s in binance_symbols if s in mapping]

    frames: dict[str, pl.DataFrame] = {c: info.candles(c, hours, as_of) for c in coins}
    panel = _panel_from_candles(frames, coins, as_of, hours)
    weights = target_weights(panel, settings.strategy, settings.risk)
    score, _ = momentum_scores(panel, settings.strategy)
    last_w = weights[-1]
    last_s = score[-1]

    positions: list[PlannedPosition] = []
    for j, coin in enumerate(coins):
        w = float(last_w[j])
        if w == 0.0 or not np.isfinite(w):
            continue
        ctx = ctxs[coin]
        notional = w * equity
        size = _round_size(abs(notional) / ctx.mark_px, ctx.sz_decimals)
        if size * ctx.mark_px < MIN_ORDER_NOTIONAL_USD:
            skipped.append(f"{coin}（最小注文額未満）")
            continue
        positions.append(
            PlannedPosition(
                binance_symbol=_binance_of(mapping, coin),
                coin=coin,
                score=float(last_s[j]) if np.isfinite(last_s[j]) else float("nan"),
                weight=w,
                notional_usd=notional,
                size=math.copysign(size, w),
                mark_px=ctx.mark_px,
                funding_8h=ctx.funding_8h,
            )
        )
    positions.sort(key=lambda p: -abs(p.weight))
    return Plan(as_of, equity, coins, positions, skipped)


def _market_of(c: AssetContext) -> HLMarket:
    return HLMarket(c.name, c.max_leverage, c.sz_decimals)


def _binance_of(mapping: dict[str, str], coin: str) -> str:
    for b, h in mapping.items():
        if h == coin:
            return b
    return coin


def _round_size(size: float, sz_decimals: int) -> float:
    step = 10.0 ** (-sz_decimals)
    return math.floor(size / step) * step


def _panel_from_candles(
    frames: dict[str, pl.DataFrame], coins: list[str], as_of: datetime, hours: int
) -> Panel:
    end = as_of.replace(minute=0, second=0, microsecond=0)
    grid = pl.datetime_range(
        end - pl.duration(hours=hours - 1),
        end,
        interval="1h",
        time_unit="ms",
        time_zone="UTC",
        eager=True,
    ).alias("open_time")
    close_cols = []
    vol_cols = []
    for c in coins:
        df = frames[c].select(
            pl.col("open_time").dt.cast_time_unit("ms"),
            "close",
            (pl.col("volume") * pl.col("close")).alias("qv"),
        )
        joined = grid.to_frame().join(df, on="open_time", how="left").sort("open_time")
        # 約定のない時間は足が来ない。値は動いていないので直前の終値で埋め、出来高は 0 にする。
        joined = joined.with_columns(
            pl.col("close").fill_null(strategy="forward"), pl.col("qv").fill_null(0.0)
        )
        close_cols.append(joined["close"].to_numpy().astype(float))
        vol_cols.append(joined["qv"].to_numpy().astype(float))
    close = np.column_stack(close_cols) if close_cols else np.zeros((grid.len(), 0))
    qv = np.column_stack(vol_cols) if vol_cols else np.zeros((grid.len(), 0))
    member = ~np.isnan(close)
    return Panel(grid.to_numpy(), coins, close, qv, np.zeros_like(close), member)
