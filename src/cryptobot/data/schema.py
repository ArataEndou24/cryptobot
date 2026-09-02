"""保存データの列定義。全ての層がこの定義を前提にする。

時刻は全て UTC。足の時刻はその足の開始時刻（open_time）で表す。
"""

from __future__ import annotations

import polars as pl

# 1時間足。Binance の配布 CSV から close_time と ignore を落としたもの。
KLINE_SCHEMA: dict[str, pl.DataType] = {
    "symbol": pl.String(),
    "open_time": pl.Datetime("ms", "UTC"),
    "open": pl.Float64(),
    "high": pl.Float64(),
    "low": pl.Float64(),
    "close": pl.Float64(),
    "volume": pl.Float64(),
    "quote_volume": pl.Float64(),
    "trades": pl.Int64(),
    "taker_buy_volume": pl.Float64(),
    "taker_buy_quote_volume": pl.Float64(),
}

# ファンディングレート（無期限先物の金利のようなもの。通常 8 時間ごと）。
FUNDING_SCHEMA: dict[str, pl.DataType] = {
    "symbol": pl.String(),
    "funding_time": pl.Datetime("ms", "UTC"),
    "funding_rate": pl.Float64(),
    "interval_hours": pl.Int32(),
}

# Binance 配布 CSV の列順（ヘッダー行の有無に関わらずこの順）。
BINANCE_KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]
BINANCE_FUNDING_COLUMNS = ["calc_time", "funding_interval_hours", "last_funding_rate"]
