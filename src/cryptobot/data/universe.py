"""ユニバース選定（対象銘柄の決め方）。

原則: 「その時点で知り得た情報だけ」で選ぶ（point-in-time）。
今の出来高上位を過去に遡って使うと、生き残った銘柄だけを選ぶ偏り（生存者バイアス）が
入り、検証結果が実態より良く見える。ここでは as_of 時点より前のデータだけで順位づけする。
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from cryptobot.config import UniverseConfig
from cryptobot.data.store import DataStore

# 集計期間に対して最低限あるべき足の割合（データ欠損の多い銘柄を避ける）。
MIN_BAR_COVERAGE = 0.90
BARS_PER_DAY = 24
# 「まだ取引されている」とみなす許容日数。配布データは 1〜2 日遅れるため余裕を持たせる。
MAX_STALENESS_DAYS = 3


def select_universe(store: DataStore, as_of: datetime, cfg: UniverseConfig) -> pl.DataFrame:
    """as_of 時点のユニバースを返す。列: rank, symbol, dollar_volume, first, last, bars."""
    if as_of.tzinfo is None:
        raise ValueError("as_of はタイムゾーン付き（UTC）の日時にしてください")
    candidates = _candidate_symbols(store, cfg)
    window_start = as_of - timedelta(days=cfg.lookback_days)
    listed_before = as_of - timedelta(days=cfg.min_history_days)
    still_trading_after = as_of - timedelta(days=MAX_STALENESS_DAYS)
    in_window = pl.col("open_time") >= window_start

    agg = (
        store.scan_klines(candidates)
        .filter(pl.col("open_time") < as_of)
        .group_by("symbol")
        .agg(
            pl.col("open_time").min().alias("first"),
            pl.col("open_time").max().alias("last"),
            pl.col("quote_volume").filter(in_window).sum().alias("dollar_volume"),
            pl.col("open_time").filter(in_window).len().alias("bars"),
        )
        .filter(
            (pl.col("first") <= listed_before)
            & (pl.col("last") >= still_trading_after)
            & (pl.col("bars") >= int(MIN_BAR_COVERAGE * cfg.lookback_days * BARS_PER_DAY))
            & (pl.col("dollar_volume") > 0)
        )
        .sort(["dollar_volume", "symbol"], descending=[True, False])
        .head(cfg.top_n)
        .with_row_index("rank", offset=1)
        .collect()
    )
    return agg.select("rank", "symbol", "dollar_volume", "first", "last", "bars")


def latest_bar_time(store: DataStore) -> datetime | None:
    """保存済みデータの最終時刻。as_of 省略時の既定値に使う。"""
    if not store.symbols():
        return None
    value = store.scan_klines().select(pl.col("open_time").max()).collect().item()
    return value if isinstance(value, datetime) else None


def _candidate_symbols(store: DataStore, cfg: UniverseConfig) -> list[str]:
    names = [s for s in store.symbols() if s.endswith(cfg.quote)]
    if cfg.include_only:
        names = [s for s in names if s in set(cfg.include_only)]
    excluded = set(cfg.exclude)
    return [s for s in names if s not in excluded]
