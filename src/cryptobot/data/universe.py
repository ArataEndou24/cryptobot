"""ユニバース選定（対象銘柄の決め方）。

原則: 「その時点で知り得た情報だけ」で選ぶ（point-in-time）。
今の出来高上位を過去に遡って使うと、生き残った銘柄だけを選ぶ偏り（生存者バイアス）が
入り、検証結果が実態より良く見える。ここでは as_of 時点より前のデータだけで順位づけする。

非暗号資産の除外:
Binance には株式、ETF、商品（金、原油）をトークン化した先物が混ざっている（2026 年以降急増）。
名前では見分けられないので、値動きの性質で判別する（2026-09-02 に実データで較正）。
- 週末の出来高が平日の 33% 未満 → 原資産が週末に休む（株、商品）。暗号資産は 40% 以上。
- 年率ボラティリティが 15% 未満 → 株価指数やステーブルコイン。暗号資産は 20% 以上。
主要な暗号資産は誤判定を避けるため名前で保護する。
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta

import polars as pl

from cryptobot.config import UniverseConfig
from cryptobot.data.store import DataStore

# 集計期間に対して最低限あるべき足の割合（データ欠損の多い銘柄を避ける）。
MIN_BAR_COVERAGE = 0.90
BARS_PER_DAY = 24
# 「まだ取引されている」とみなす許容日数。配布データは 1〜2 日遅れるため余裕を持たせる。
MAX_STALENESS_DAYS = 3

# 非暗号資産の判定（研究記録 2026-09-02 の較正表を参照）
CLASSIFY_LOOKBACK_DAYS = 60
MIN_CLASSIFY_DAYS = 28
NON_CRYPTO_MAX_WEEKEND_VOLUME_RATIO = 0.33
MIN_ANNUAL_VOL = 0.15
ALWAYS_CRYPTO: frozenset[str] = frozenset(
    {
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "XRPUSDT",
        "BNBUSDT",
        "DOGEUSDT",
        "ADAUSDT",
        "LTCUSDT",
        "BCHUSDT",
        "LINKUSDT",
        "AVAXUSDT",
        "DOTUSDT",
        "TRXUSDT",
    }
)

DAILY_SCHEMA: dict[str, pl.DataType] = {
    "symbol": pl.String(),
    "day": pl.Date(),
    "dollar_volume": pl.Float64(),
    "bars": pl.UInt32(),
    "close": pl.Float64(),
    "weekend": pl.Boolean(),
}


def compute_daily(store: DataStore, symbols: Iterable[str] | None = None) -> pl.DataFrame:
    """銘柄 × 日の集計表。ユニバース選定はこの表だけを使う（1 時間足を何度も読まないため）。"""
    names = list(symbols) if symbols is not None else store.symbols()
    if not names:
        return pl.DataFrame(schema=DAILY_SCHEMA)
    return (
        store.scan_klines(names)
        .sort(["symbol", "open_time"])
        .group_by_dynamic("open_time", every="1d", group_by="symbol")
        .agg(
            pl.col("quote_volume").sum().alias("dollar_volume"),
            pl.len().cast(pl.UInt32).alias("bars"),
            pl.col("close").last().alias("close"),
        )
        .with_columns(
            pl.col("open_time").dt.date().alias("day"),
            (pl.col("open_time").dt.weekday() >= 6).alias("weekend"),
        )
        .select(list(DAILY_SCHEMA))
        .collect()
    )


def classify(daily: pl.DataFrame, as_of: datetime) -> pl.DataFrame:
    """as_of 直前 60 日の性質から暗号資産かどうかを判定する。

    列: symbol, weekend_volume_ratio, annual_vol, days, is_crypto, reason
    """
    lo = (as_of - timedelta(days=CLASSIFY_LOOKBACK_DAYS)).date()
    hi = as_of.date()
    w = daily.filter((pl.col("day") >= lo) & (pl.col("day") < hi)).sort(["symbol", "day"])
    stats = w.group_by("symbol").agg(
        (
            pl.col("dollar_volume").filter(pl.col("weekend")).mean()
            / pl.col("dollar_volume").filter(~pl.col("weekend")).mean()
        ).alias("weekend_volume_ratio"),
        (pl.col("close").log().diff().std() * (365.0**0.5)).alias("annual_vol"),
        pl.len().alias("days"),
    )
    protected = pl.col("symbol").is_in(list(ALWAYS_CRYPTO))
    enough = pl.col("days") >= MIN_CLASSIFY_DAYS
    weekend_quiet = pl.col("weekend_volume_ratio") < NON_CRYPTO_MAX_WEEKEND_VOLUME_RATIO
    low_vol = pl.col("annual_vol") < MIN_ANNUAL_VOL
    return stats.with_columns(
        pl.when(protected)
        .then(pl.lit("主要暗号資産"))
        .when(~enough)
        .then(pl.lit("判定に必要な日数が足りない"))
        .when(low_vol)
        .then(pl.lit("値動きが小さすぎる（指数・ステーブルコイン）"))
        .when(weekend_quiet)
        .then(pl.lit("週末に休む（株・商品）"))
        .otherwise(pl.lit("暗号資産"))
        .alias("reason")
    ).with_columns((protected | (enough & ~low_vol & ~weekend_quiet)).alias("is_crypto"))


def select_universe_from_daily(
    daily: pl.DataFrame, as_of: datetime, cfg: UniverseConfig
) -> pl.DataFrame:
    """as_of 時点のユニバース。列: rank, symbol, dollar_volume, first, last, bars."""
    if as_of.tzinfo is None:
        raise ValueError("as_of はタイムゾーン付き（UTC）の日時にしてください")
    hi = as_of.date()
    window_start = (as_of - timedelta(days=cfg.lookback_days)).date()
    listed_before = (as_of - timedelta(days=cfg.min_history_days)).date()
    still_trading_after = (as_of - timedelta(days=MAX_STALENESS_DAYS)).date()
    in_window = pl.col("day") >= window_start

    cand = daily.filter(pl.col("symbol").is_in(_candidate_names(daily, cfg)))
    agg = (
        cand.filter(pl.col("day") < hi)
        .group_by("symbol")
        .agg(
            pl.col("day").min().alias("first"),
            pl.col("day").max().alias("last"),
            pl.col("dollar_volume").filter(in_window).sum().alias("dollar_volume"),
            pl.col("bars").filter(in_window).sum().alias("bars"),
        )
        .filter(
            (pl.col("first") <= listed_before)
            & (pl.col("last") >= still_trading_after)
            & (pl.col("bars") >= int(MIN_BAR_COVERAGE * cfg.lookback_days * BARS_PER_DAY))
            & (pl.col("dollar_volume") > 0)
        )
    )
    if cfg.exclude_non_crypto and not agg.is_empty():
        cls = classify(cand.filter(pl.col("symbol").is_in(agg["symbol"].to_list())), as_of)
        agg = agg.join(cls.select("symbol", "is_crypto"), on="symbol", how="left").filter(
            pl.col("is_crypto").fill_null(False)
        )
    return (
        agg.sort(["dollar_volume", "symbol"], descending=[True, False])
        .head(cfg.top_n)
        .with_row_index("rank", offset=1)
        .select("rank", "symbol", "dollar_volume", "first", "last", "bars")
    )


def select_universe(store: DataStore, as_of: datetime, cfg: UniverseConfig) -> pl.DataFrame:
    return select_universe_from_daily(compute_daily(store), as_of, cfg)


def excluded_non_crypto(daily: pl.DataFrame, as_of: datetime, cfg: UniverseConfig) -> pl.DataFrame:
    """as_of 時点で「暗号資産でない」と判定された銘柄の一覧（運用者が目で確かめるため）。"""
    cand = daily.filter(pl.col("symbol").is_in(_candidate_names(daily, cfg)))
    return (
        classify(cand, as_of)
        .filter(~pl.col("is_crypto") & pl.col("weekend_volume_ratio").fill_nan(None).is_not_null())
        .sort("weekend_volume_ratio")
        .select("symbol", "weekend_volume_ratio", "annual_vol", "days", "reason")
    )


def latest_bar_time(store: DataStore) -> datetime | None:
    """保存済みデータの最終時刻。as_of 省略時の既定値に使う。"""
    if not store.symbols():
        return None
    value = store.scan_klines().select(pl.col("open_time").max()).collect().item()
    return value if isinstance(value, datetime) else None


def _candidate_names(daily: pl.DataFrame, cfg: UniverseConfig) -> list[str]:
    names = [s for s in daily["symbol"].unique().to_list() if s.endswith(cfg.quote)]
    if cfg.include_only:
        names = [s for s in names if s in set(cfg.include_only)]
    if cfg.tradable_only and TRADABLE is not None:
        names = [s for s in names if s in TRADABLE]
    excluded = set(cfg.exclude)
    return sorted(s for s in names if s not in excluded)


# 取引所で取引可能な Binance 銘柄名の集合。set_tradable() で設定する。None なら絞り込みなし。
TRADABLE: frozenset[str] | None = None


def set_tradable(symbols: Iterable[str] | None) -> None:
    global TRADABLE
    TRADABLE = None if symbols is None else frozenset(symbols)
