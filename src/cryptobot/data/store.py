"""Parquet 形式のローカルデータストア。

配置:
  <root>/raw/binance_um/...          取得した zip をそのまま保存（再取得を避けるため）
  <root>/parquet/binance_um/klines_1h/<SYMBOL>.parquet
  <root>/parquet/binance_um/funding/<SYMBOL>.parquet
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import polars as pl

from cryptobot.data.schema import FUNDING_SCHEMA, KLINE_SCHEMA

SOURCE = "binance_um"


class DataStore:
    def __init__(self, root: Path, interval: str = "1h") -> None:
        self.root = root
        self.interval = interval

    # ---- ディレクトリ ----
    @property
    def raw_dir(self) -> Path:
        return self.root / "raw" / SOURCE

    @property
    def klines_dir(self) -> Path:
        return self.root / "parquet" / SOURCE / f"klines_{self.interval}"

    @property
    def funding_dir(self) -> Path:
        return self.root / "parquet" / SOURCE / "funding"

    def klines_path(self, symbol: str) -> Path:
        return self.klines_dir / f"{symbol}.parquet"

    def funding_path(self, symbol: str) -> Path:
        return self.funding_dir / f"{symbol}.parquet"

    # ---- 書き込み ----
    def write_klines(self, symbol: str, df: pl.DataFrame) -> Path:
        df = _conform(df, KLINE_SCHEMA).unique(subset=["open_time"], keep="last").sort("open_time")
        path = self.klines_path(symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path, compression="zstd")
        return path

    def write_funding(self, symbol: str, df: pl.DataFrame) -> Path:
        df = (
            _conform(df, FUNDING_SCHEMA)
            .unique(subset=["funding_time"], keep="last")
            .sort("funding_time")
        )
        path = self.funding_path(symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path, compression="zstd")
        return path

    # ---- 読み込み ----
    def symbols(self) -> list[str]:
        if not self.klines_dir.exists():
            return []
        return sorted(p.stem for p in self.klines_dir.glob("*.parquet"))

    def scan_klines(self, symbols: Iterable[str] | None = None) -> pl.LazyFrame:
        names = list(symbols) if symbols is not None else self.symbols()
        paths = [self.klines_path(s) for s in names if self.klines_path(s).exists()]
        if not paths:
            return pl.LazyFrame(schema=KLINE_SCHEMA)
        return pl.scan_parquet(paths)

    def scan_funding(self, symbols: Iterable[str] | None = None) -> pl.LazyFrame:
        names = list(symbols) if symbols is not None else self.symbols()
        paths = [self.funding_path(s) for s in names if self.funding_path(s).exists()]
        if not paths:
            return pl.LazyFrame(schema=FUNDING_SCHEMA)
        return pl.scan_parquet(paths)

    def read_klines(self, symbol: str) -> pl.DataFrame:
        return pl.read_parquet(self.klines_path(symbol))

    def read_funding(self, symbol: str) -> pl.DataFrame:
        return pl.read_parquet(self.funding_path(symbol))

    def summary(self) -> pl.DataFrame:
        """銘柄ごとの保有期間と行数。`data status` コマンドで使う。"""
        if not self.symbols():
            return pl.DataFrame(
                schema={
                    "symbol": pl.String(),
                    "first": pl.Datetime("ms", "UTC"),
                    "last": pl.Datetime("ms", "UTC"),
                    "rows": pl.UInt32(),
                }
            )
        return (
            self.scan_klines()
            .group_by("symbol")
            .agg(
                pl.col("open_time").min().alias("first"),
                pl.col("open_time").max().alias("last"),
                pl.len().alias("rows"),
            )
            .sort("symbol")
            .collect()
        )


def _conform(df: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    missing = [c for c in schema if c not in df.columns]
    if missing:
        raise ValueError(f"必要な列がありません: {missing}")
    return df.select([pl.col(c).cast(t) for c, t in schema.items()])
