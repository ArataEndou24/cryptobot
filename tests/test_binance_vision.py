from __future__ import annotations

from datetime import date

import polars as pl

from cryptobot.data import binance_vision as bv
from tests.conftest import make_zip

KLINE_ROW_OLD = "1577836800000,7189.43,7190.52,7170.15,7171.55,2449.049,1577840399999,17576424.4,3688,996.198,7149370.7,0\n"
KLINE_HEADER = "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore\n"
KLINE_ROW_NEW = "1748736000000,104544.60,104599.60,104277.30,104408.70,3438.526,1748739599999,358998670.7,71534,1691.726,176614558.1,0\n"
KLINE_ROW_US = "1748739600000000,104408.70,104500.00,104300.00,104450.00,1000.0,1748743199999999,100000000.0,500,500.0,50000000.0,0\n"


def test_parse_kline_without_header() -> None:
    df = bv.parse_kline_csv(KLINE_ROW_OLD.encode(), "BTCUSDT")
    assert df.height == 1
    assert df["symbol"][0] == "BTCUSDT"
    assert str(df["open_time"][0]) == "2020-01-01 00:00:00+00:00"
    assert df["close"][0] == 7171.55
    assert df["trades"][0] == 3688
    assert "close_time" not in df.columns


def test_parse_kline_with_header_and_microseconds() -> None:
    raw = (KLINE_HEADER + KLINE_ROW_NEW + KLINE_ROW_US).encode()
    df = bv.parse_kline_csv(raw, "BTCUSDT")
    assert df.height == 2
    assert str(df["open_time"][0]) == "2025-06-01 00:00:00+00:00"
    assert str(df["open_time"][1]) == "2025-06-01 01:00:00+00:00"
    assert df.schema["open_time"] == pl.Datetime("ms", "UTC")


def test_parse_funding_rounds_to_second() -> None:
    raw = (
        b"calc_time,funding_interval_hours,last_funding_rate\n"
        b"1748736000001,8,-0.00000582\n1748764800002,8,0.00002335\n"
    )
    df = bv.parse_funding_csv(raw, "BTCUSDT")
    assert df.height == 2
    assert str(df["funding_time"][0]) == "2025-06-01 00:00:00+00:00"
    assert df["funding_rate"][1] == 0.00002335
    assert df["interval_hours"][0] == 8


def test_parse_zip_roundtrip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    p = tmp_path / "x.zip"
    p.write_bytes(make_zip(KLINE_HEADER + KLINE_ROW_NEW, "BTCUSDT-1h-2025-06.csv"))
    df = bv.parse_kline_zip(p, "BTCUSDT")
    assert df.height == 1


def test_month_range() -> None:
    assert bv.month_range("2023-11", date(2024, 2, 15)) == [
        "2023-11",
        "2023-12",
        "2024-01",
        "2024-02",
    ]
    assert bv.month_range("2024-03", date(2024, 2, 15)) == []


def test_key_helpers() -> None:
    k = bv.monthly_kline_key("BTCUSDT", "1h", "2024-01")
    assert k == "data/futures/um/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2024-01.zip"
    assert bv.month_of_key(k) == "2024-01"
    d = bv.daily_kline_key("BTCUSDT", "1h", date(2024, 1, 5))
    assert bv.day_of_key(d) == date(2024, 1, 5)
    assert bv.month_of_key(d) is None


def test_filter_perp_symbols() -> None:
    raw = ["BTCUSDT", "BTCUSDT_210326", "ETHBUSD", "1000PEPEUSDT", "ETHUSDC", "btcusdt"]
    assert bv.filter_perp_symbols(raw) == ["1000PEPEUSDT", "BTCUSDT"]
    assert bv.filter_perp_symbols(raw, "USDC") == ["ETHUSDC"]


def test_count_gaps() -> None:
    s = pl.Series(
        "t",
        ["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 03:00", "2024-01-01 04:00"],
    ).str.to_datetime("%Y-%m-%d %H:%M")
    assert bv._count_gaps(s, "1h") == 1
