"""Binance 公式の過去データ配布（data.binance.vision）からの取得。

対象は USDT 建て無期限先物（USDⓈ-M, "um"）の 1 時間足とファンディングレート。
口座も API キーも不要。研究用データとしてはこれが最も質が高く無料である。

配布の仕様（2026-09 時点で確認済み）:
- 月次ファイル: data/futures/um/monthly/klines/<SYMBOL>/<interval>/<SYMBOL>-<interval>-YYYY-MM.zip
- 日次ファイル: data/futures/um/daily/klines/<SYMBOL>/<interval>/<SYMBOL>-<interval>-YYYY-MM-DD.zip
- ファンディング: data/futures/um/monthly/fundingRate/<SYMBOL>/<SYMBOL>-fundingRate-YYYY-MM.zip
- 各 zip の隣に <name>.zip.CHECKSUM（sha256）がある。
- 月次ファイルは月末から数日遅れて出る。それまでは日次ファイルで補う。
- 古い CSV にはヘッダー行がなく、新しい CSV にはある。列順は同じ。
- 時刻はミリ秒。将来マイクロ秒に変わる可能性があるため両方を受け付ける。
"""

from __future__ import annotations

import hashlib
import io
import re
import time
import zipfile
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from xml.etree import ElementTree

import httpx
import polars as pl

from cryptobot.data.schema import BINANCE_FUNDING_COLUMNS, BINANCE_KLINE_COLUMNS
from cryptobot.data.store import DataStore

DATA_BASE = "https://data.binance.vision/"
S3_LIST_BASE = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
UM_PREFIX = "data/futures/um"
_S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
_MONTH_RE = re.compile(r"-(\d{4}-\d{2})\.zip$")
_DAY_RE = re.compile(r"-(\d{4}-\d{2}-\d{2})\.zip$")


def make_client(timeout: float = 60.0) -> httpx.Client:
    return httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": "cryptobot/0.1 (research data sync)"},
    )


# ---------------------------------------------------------------------------
# 一覧取得
# ---------------------------------------------------------------------------


def list_keys(client: httpx.Client, prefix: str, delimiter: str | None = None) -> list[str]:
    """S3 の一覧 API でキー（または delimiter 指定時は共通プレフィックス）を全件取る。"""
    results: list[str] = []
    marker: str | None = None
    while True:
        params: dict[str, str] = {"prefix": prefix}
        if delimiter:
            params["delimiter"] = delimiter
        if marker:
            params["marker"] = marker
        resp = _get_with_retry(client, S3_LIST_BASE, params=params)
        root = ElementTree.fromstring(resp.content)
        if delimiter:
            for cp in root.iter(f"{_S3_NS}CommonPrefixes"):
                p = cp.find(f"{_S3_NS}Prefix")
                if p is not None and p.text:
                    results.append(p.text)
        last_key: str | None = None
        for c in root.iter(f"{_S3_NS}Contents"):
            k = c.find(f"{_S3_NS}Key")
            if k is not None and k.text:
                last_key = k.text
                if not delimiter:
                    results.append(k.text)
        truncated = (root.findtext(f"{_S3_NS}IsTruncated") or "false").lower() == "true"
        if not truncated:
            break
        marker = root.findtext(f"{_S3_NS}NextMarker") or last_key
        if marker is None:
            break
    return results


def list_symbols(client: httpx.Client) -> list[str]:
    """配布されている全銘柄（USDⓈ-M 先物）。上場廃止済みの銘柄も含む。"""
    prefixes = list_keys(client, f"{UM_PREFIX}/monthly/klines/", delimiter="/")
    return sorted(p.rstrip("/").rsplit("/", 1)[-1] for p in prefixes)


def filter_perp_symbols(symbols: Iterable[str], quote: str = "USDT") -> list[str]:
    """無期限先物のみ残す。"BTCUSDT_210326" のような期限付き先物は除く。"""
    pattern = re.compile(rf"^[A-Z0-9]+{re.escape(quote)}$")
    return sorted(s for s in symbols if pattern.match(s))


# ---------------------------------------------------------------------------
# 取得計画
# ---------------------------------------------------------------------------


def month_range(start_month: str, end: date) -> list[str]:
    """start_month（YYYY-MM）から end の月までの年月一覧。"""
    y, m = (int(x) for x in start_month.split("-"))
    out: list[str] = []
    while (y, m) <= (end.year, end.month):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def month_of_key(key: str) -> str | None:
    mt = _MONTH_RE.search(key)
    return mt.group(1) if mt else None


def day_of_key(key: str) -> date | None:
    mt = _DAY_RE.search(key)
    return date.fromisoformat(mt.group(1)) if mt else None


def monthly_kline_key(symbol: str, interval: str, month: str) -> str:
    return f"{UM_PREFIX}/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{month}.zip"


def daily_kline_key(symbol: str, interval: str, day: date) -> str:
    return f"{UM_PREFIX}/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{day.isoformat()}.zip"


def monthly_funding_key(symbol: str, month: str) -> str:
    return f"{UM_PREFIX}/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{month}.zip"


@dataclass(frozen=True)
class SyncPlan:
    symbol: str
    monthly_klines: list[str]
    daily_klines: list[str]
    monthly_funding: list[str]

    @property
    def all_keys(self) -> list[str]:
        return [*self.monthly_klines, *self.daily_klines, *self.monthly_funding]


def plan_symbol(
    client: httpx.Client,
    symbol: str,
    interval: str,
    start_month: str,
    today: date,
    include_funding: bool = True,
) -> SyncPlan:
    """その銘柄で取得すべきファイルの一覧を作る。

    月次で存在するものは月次を使い、直近の月次未公開の月だけ日次で補う。
    上場前の月は配布側に存在しないので、自然に対象外になる。
    """
    monthly = [
        k
        for k in list_keys(client, f"{UM_PREFIX}/monthly/klines/{symbol}/{interval}/")
        if k.endswith(".zip") and (month_of_key(k) or "") >= start_month
    ]
    have_months = {month_of_key(k) for k in monthly}
    # 月次が出ていない直近の月（最大 2 か月）は日次で補う。
    recent_months = month_range(start_month, today)[-2:]
    daily: list[str] = []
    yesterday = today - timedelta(days=1)
    for month in recent_months:
        if month in have_months:
            continue
        y, m = (int(x) for x in month.split("-"))
        d = date(y, m, 1)
        while d.month == m and d <= yesterday:
            daily.append(daily_kline_key(symbol, interval, d))
            d += timedelta(days=1)
    funding: list[str] = []
    if include_funding:
        funding = [
            k
            for k in list_keys(client, f"{UM_PREFIX}/monthly/fundingRate/{symbol}/")
            if k.endswith(".zip") and (month_of_key(k) or "") >= start_month
        ]
    return SyncPlan(symbol, monthly, daily, funding)


# ---------------------------------------------------------------------------
# ダウンロード
# ---------------------------------------------------------------------------


class DownloadError(Exception):
    pass


def _get_with_retry(
    client: httpx.Client,
    url: str,
    params: dict[str, str] | None = None,
    attempts: int = 4,
) -> httpx.Response:
    delay = 1.0
    last: Exception | None = None
    for _ in range(attempts):
        try:
            resp = client.get(url, params=params)
            if resp.status_code == 404:
                return resp
            if resp.status_code >= 500 or resp.status_code == 429:
                raise DownloadError(f"HTTP {resp.status_code}: {url}")
            resp.raise_for_status()
            return resp
        except (httpx.HTTPError, DownloadError) as e:
            last = e
            time.sleep(delay)
            delay *= 2
    raise DownloadError(f"取得に失敗しました（{attempts} 回試行）: {url}\n  {last}")


def raw_path_for(store: DataStore, key: str) -> Path:
    """zip の保存先。配布側のパス構造をそのまま raw/ 以下に写す。"""
    rel = key.removeprefix(UM_PREFIX + "/")
    return store.raw_dir / rel


def download_key(
    client: httpx.Client, store: DataStore, key: str, verify_checksum: bool = True
) -> Path | None:
    """1 ファイルを取得して保存する。既に存在すれば取得しない。404 なら None。"""
    dest = raw_path_for(store, key)
    if dest.exists():
        return dest
    resp = _get_with_retry(client, DATA_BASE + key)
    if resp.status_code == 404:
        return None
    body = resp.content
    if verify_checksum:
        cs = _get_with_retry(client, DATA_BASE + key + ".CHECKSUM")
        if cs.status_code != 404:
            expected = cs.text.split()[0].strip().lower()
            actual = hashlib.sha256(body).hexdigest()
            if expected != actual:
                raise DownloadError(f"チェックサム不一致（ファイル破損の可能性）: {key}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(body)
    tmp.replace(dest)
    return dest


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------


def _read_zip_csv(path: Path) -> bytes:
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"zip 内の CSV が 1 つではありません: {path} -> {names}")
        return zf.read(names[0])


def _has_header(raw: bytes, first_column: str) -> bool:
    head = raw[:64].lstrip().lower()
    return head.startswith(first_column.encode())


def _normalize_epoch(col: str) -> pl.Expr:
    """ミリ秒/マイクロ秒どちらで来てもミリ秒に揃える。"""
    c = pl.col(col)
    return pl.when(c > 10**14).then(c // 1000).otherwise(c).alias(col)


def parse_kline_csv(raw: bytes, symbol: str) -> pl.DataFrame:
    has_header = _has_header(raw, "open_time")
    df = pl.read_csv(
        io.BytesIO(raw),
        has_header=False,
        skip_rows=1 if has_header else 0,
        new_columns=BINANCE_KLINE_COLUMNS,
        schema_overrides={
            "open_time": pl.Int64,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
            "close_time": pl.Int64,
            "quote_volume": pl.Float64,
            "count": pl.Int64,
            "taker_buy_volume": pl.Float64,
            "taker_buy_quote_volume": pl.Float64,
            "ignore": pl.String,
        },
    )
    return df.select(
        pl.lit(symbol).alias("symbol"),
        pl.from_epoch(_normalize_epoch("open_time"), time_unit="ms")
        .dt.cast_time_unit("ms")
        .dt.replace_time_zone("UTC")
        .alias("open_time"),
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        pl.col("count").alias("trades"),
        "taker_buy_volume",
        "taker_buy_quote_volume",
    )


def parse_funding_csv(raw: bytes, symbol: str) -> pl.DataFrame:
    has_header = _has_header(raw, "calc_time")
    df = pl.read_csv(
        io.BytesIO(raw),
        has_header=False,
        skip_rows=1 if has_header else 0,
        new_columns=BINANCE_FUNDING_COLUMNS,
        schema_overrides={
            "calc_time": pl.Int64,
            "funding_interval_hours": pl.Int32,
            "last_funding_rate": pl.Float64,
        },
    )
    # calc_time は 8 時間境界から数ミリ秒ずれて記録される。足に合わせて秒単位に丸める。
    return df.select(
        pl.lit(symbol).alias("symbol"),
        pl.from_epoch((_normalize_epoch("calc_time") // 1000) * 1000, time_unit="ms")
        .dt.cast_time_unit("ms")
        .dt.replace_time_zone("UTC")
        .alias("funding_time"),
        pl.col("last_funding_rate").alias("funding_rate"),
        pl.col("funding_interval_hours").alias("interval_hours"),
    )


def parse_kline_zip(path: Path, symbol: str) -> pl.DataFrame:
    return parse_kline_csv(_read_zip_csv(path), symbol)


def parse_funding_zip(path: Path, symbol: str) -> pl.DataFrame:
    return parse_funding_csv(_read_zip_csv(path), symbol)


# ---------------------------------------------------------------------------
# 同期（計画 → 取得 → 解析 → Parquet）
# ---------------------------------------------------------------------------


@dataclass
class SyncResult:
    symbol: str
    files_downloaded: int = 0
    files_total: int = 0
    kline_rows: int = 0
    funding_rows: int = 0
    first: datetime | None = None
    last: datetime | None = None
    error: str | None = None
    warnings: list[str] = field(default_factory=list)


def sync_symbol(
    client: httpx.Client,
    store: DataStore,
    symbol: str,
    interval: str,
    start_month: str,
    today: date | None = None,
    verify_checksum: bool = True,
    include_funding: bool = True,
) -> SyncResult:
    today = today or datetime.now(UTC).date()
    result = SyncResult(symbol=symbol)
    try:
        plan = plan_symbol(client, symbol, interval, start_month, today, include_funding)
        result.files_total = len(plan.all_keys)
        kline_paths: list[Path] = []
        funding_paths: list[Path] = []
        for key in plan.monthly_klines + plan.daily_klines:
            existed = raw_path_for(store, key).exists()
            p = download_key(client, store, key, verify_checksum)
            if p is None:
                continue
            if not existed:
                result.files_downloaded += 1
            kline_paths.append(p)
        for key in plan.monthly_funding:
            existed = raw_path_for(store, key).exists()
            p = download_key(client, store, key, verify_checksum)
            if p is None:
                continue
            if not existed:
                result.files_downloaded += 1
            funding_paths.append(p)

        if kline_paths:
            frames = [parse_kline_zip(p, symbol) for p in kline_paths]
            df = pl.concat(frames, how="vertical")
            store.write_klines(symbol, df)
            written = store.read_klines(symbol)
            result.kline_rows = written.height
            result.first = written["open_time"].min()  # type: ignore[assignment]
            result.last = written["open_time"].max()  # type: ignore[assignment]
            gaps = _count_gaps(written["open_time"], interval)
            if gaps:
                result.warnings.append(f"欠損している足が {gaps} 本あります")
        else:
            result.warnings.append("配布データがありません（上場前、または取り扱い外）")
        if funding_paths:
            fdf = pl.concat([parse_funding_zip(p, symbol) for p in funding_paths], how="vertical")
            store.write_funding(symbol, fdf)
            result.funding_rows = store.read_funding(symbol).height
    except Exception as e:  # 1 銘柄の失敗で全体を止めない
        result.error = f"{type(e).__name__}: {e}"
    return result


def _count_gaps(times: pl.Series, interval: str) -> int:
    step = {"1h": timedelta(hours=1)}[interval]
    if times.len() < 2:
        return 0
    diffs = times.sort().diff().drop_nulls()
    return int((diffs > step).sum())


def sync_many(
    client: httpx.Client,
    store: DataStore,
    symbols: Iterable[str],
    interval: str,
    start_month: str,
    workers: int = 8,
    verify_checksum: bool = True,
    include_funding: bool = True,
    on_result: Callable[[SyncResult], None] | None = None,
) -> list[SyncResult]:
    names = list(symbols)
    results: list[SyncResult] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(
                sync_symbol,
                client,
                store,
                s,
                interval,
                start_month,
                None,
                verify_checksum,
                include_funding,
            ): s
            for s in names
        }
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            if on_result:
                on_result(r)
    results.sort(key=lambda r: r.symbol)
    return results
