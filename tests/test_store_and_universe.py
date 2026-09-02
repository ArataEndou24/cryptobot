from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from cryptobot.config import UniverseConfig
from cryptobot.data.store import DataStore
from cryptobot.data.universe import select_universe
from tests.conftest import synthetic_klines


def test_store_dedups_and_sorts(store: DataStore, t0: datetime) -> None:
    a = synthetic_klines("BTCUSDT", t0, 5)
    b = synthetic_klines("BTCUSDT", t0 + timedelta(hours=3), 5, price=200.0)
    store.write_klines("BTCUSDT", pl.concat([b, a]))
    df = store.read_klines("BTCUSDT")
    assert df.height == 8  # 0..4 と 3..7 の和集合
    assert df["open_time"].is_sorted()
    # 重複した足は後勝ち（新しいデータで上書き）
    assert df.filter(pl.col("open_time") == t0 + timedelta(hours=3))["close"][0] == 100.0
    assert store.symbols() == ["BTCUSDT"]
    assert store.summary()["rows"][0] == 8


def test_store_rejects_missing_columns(store: DataStore) -> None:
    with pytest.raises(ValueError):
        store.write_klines("X", pl.DataFrame({"symbol": ["X"]}))


def _cfg(**kw: object) -> UniverseConfig:
    base = {"top_n": 2, "lookback_days": 2, "min_history_days": 5}
    base.update(kw)
    return UniverseConfig.model_validate(base)


def test_universe_ranks_by_dollar_volume_point_in_time(store: DataStore, t0: datetime) -> None:
    days = 10
    store.write_klines("AAAUSDT", synthetic_klines("AAAUSDT", t0, days * 24, quote_volume=1e6))
    store.write_klines("BBBUSDT", synthetic_klines("BBBUSDT", t0, days * 24, quote_volume=3e6))
    store.write_klines("CCCUSDT", synthetic_klines("CCCUSDT", t0, days * 24, quote_volume=2e6))
    # 上場が新しすぎる銘柄（min_history_days 未満）
    store.write_klines(
        "NEWUSDT", synthetic_klines("NEWUSDT", t0 + timedelta(days=8), 2 * 24, quote_volume=9e6)
    )
    # 上場廃止済み（as_of の 3 日以上前に止まっている）
    store.write_klines("DEADUSDT", synthetic_klines("DEADUSDT", t0, 6 * 24, quote_volume=9e6))
    # 決済通貨が違う
    store.write_klines("ETHUSDC", synthetic_klines("ETHUSDC", t0, days * 24, quote_volume=9e6))

    as_of = t0 + timedelta(days=days)
    u = select_universe(store, as_of, _cfg())
    assert u["symbol"].to_list() == ["BBBUSDT", "CCCUSDT"]
    assert u["rank"].to_list() == [1, 2]
    assert u["dollar_volume"][0] == pytest.approx(3e6 * 48)


def test_universe_exclude_and_include_only(store: DataStore, t0: datetime) -> None:
    for s, qv in [("AAAUSDT", 1e6), ("BBBUSDT", 3e6), ("CCCUSDT", 2e6)]:
        store.write_klines(s, synthetic_klines(s, t0, 10 * 24, quote_volume=qv))
    as_of = t0 + timedelta(days=10)
    assert select_universe(store, as_of, _cfg(exclude=["BBBUSDT"]))["symbol"].to_list() == [
        "CCCUSDT",
        "AAAUSDT",
    ]
    assert select_universe(store, as_of, _cfg(include_only=["AAAUSDT"]))["symbol"].to_list() == [
        "AAAUSDT"
    ]


def test_universe_uses_only_past_data(store: DataStore, t0: datetime) -> None:
    # as_of 以降に出来高が急増しても、as_of 時点の順位には影響しない
    a = synthetic_klines("AAAUSDT", t0, 10 * 24, quote_volume=1e6)
    b_past = synthetic_klines("BBBUSDT", t0, 10 * 24, quote_volume=2e6)
    b_future = synthetic_klines("BBBUSDT", t0 + timedelta(days=10), 24, quote_volume=1e9)
    a_future = synthetic_klines("AAAUSDT", t0 + timedelta(days=10), 24, quote_volume=1e12)
    store.write_klines("AAAUSDT", pl.concat([a, a_future]))
    store.write_klines("BBBUSDT", pl.concat([b_past, b_future]))
    u = select_universe(store, t0 + timedelta(days=10), _cfg())
    assert u["symbol"].to_list() == ["BBBUSDT", "AAAUSDT"]


def test_universe_requires_tz(store: DataStore) -> None:
    with pytest.raises(ValueError):
        select_universe(store, datetime(2024, 1, 1), _cfg())


def test_universe_empty_store(store: DataStore) -> None:
    assert select_universe(store, datetime(2024, 1, 1, tzinfo=UTC), _cfg()).is_empty()
