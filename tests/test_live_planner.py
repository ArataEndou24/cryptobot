from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from cryptobot.exchange.hyperliquid_info import account_equity, open_positions, parse_candles
from cryptobot.live.planner import _panel_from_candles, _round_size


def test_parse_candles() -> None:
    raw = [
        {
            "t": 1788339600000,
            "T": 1,
            "s": "BTC",
            "i": "1h",
            "o": "1",
            "c": "2",
            "h": "3",
            "l": "0.5",
            "v": "10",
            "n": 5,
        },
        {
            "t": 1788336000000,
            "T": 1,
            "s": "BTC",
            "i": "1h",
            "o": "1",
            "c": "1.5",
            "h": "3",
            "l": "0.5",
            "v": "10",
            "n": 5,
        },
    ]
    df = parse_candles(raw)
    assert df["open_time"].is_sorted()
    assert df["close"].to_list() == [1.5, 2.0]
    assert parse_candles([]).is_empty()


def test_account_helpers() -> None:
    state = {
        "marginSummary": {"accountValue": "1234.5"},
        "assetPositions": [
            {"position": {"coin": "BTC", "szi": "0.01"}},
            {"position": {"coin": "ETH", "szi": "-1.5"}},
            {"position": {"coin": "SOL", "szi": "0.0"}},
        ],
    }
    assert account_equity(state) == 1234.5
    assert open_positions(state) == {"BTC": 0.01, "ETH": -1.5}


def test_round_size() -> None:
    assert _round_size(0.123456, 3) == 0.123
    assert _round_size(12.7, 0) == 12.0


def test_panel_from_candles_aligns_to_grid() -> None:
    as_of = datetime(2026, 9, 2, 10, 30, tzinfo=UTC)
    hours = 5
    end = as_of.replace(minute=0)
    times = [end - timedelta(hours=i) for i in (4, 3, 1, 0)]  # 2 時間前が欠けている
    df = pl.DataFrame(
        {"open_time": times, "close": [1.0, 2.0, 4.0, 5.0], "volume": [1.0] * 4}
    ).with_columns(pl.col("open_time").dt.cast_time_unit("ms"))
    p = _panel_from_candles({"BTC": df}, ["BTC"], as_of, hours)
    assert p.n_times == hours and p.symbols == ["BTC"]
    # 欠けた時間は直前の終値で埋まる（値が動いていない扱い）
    assert p.close[2, 0] == 2.0 and p.close[-1, 0] == 5.0
    assert p.quote_volume[2, 0] == 0.0 and p.member[-1, 0]
