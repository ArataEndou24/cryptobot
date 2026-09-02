from __future__ import annotations

import io
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from cryptobot.data.store import DataStore


def make_zip(csv_text: str, inner_name: str = "x.csv") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(inner_name, csv_text)
    return buf.getvalue()


def synthetic_klines(
    symbol: str, start: datetime, hours: int, price: float = 100.0, quote_volume: float = 1e6
) -> pl.DataFrame:
    times = [start + timedelta(hours=i) for i in range(hours)]
    return pl.DataFrame(
        {
            "symbol": [symbol] * hours,
            "open_time": times,
            "open": [price] * hours,
            "high": [price * 1.01] * hours,
            "low": [price * 0.99] * hours,
            "close": [price] * hours,
            "volume": [quote_volume / price] * hours,
            "quote_volume": [quote_volume] * hours,
            "trades": [10] * hours,
            "taker_buy_volume": [quote_volume / price / 2] * hours,
            "taker_buy_quote_volume": [quote_volume / 2] * hours,
        }
    ).with_columns(pl.col("open_time").dt.replace_time_zone("UTC").dt.cast_time_unit("ms"))


@pytest.fixture
def store(tmp_path: Path) -> DataStore:
    return DataStore(tmp_path / "data", "1h")


@pytest.fixture
def t0() -> datetime:
    return datetime(2024, 1, 1, tzinfo=UTC)
