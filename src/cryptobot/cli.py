"""運用者向けコマンド。`cryptobot --help` で一覧が出る。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated

import polars as pl
import typer

from cryptobot import __version__
from cryptobot.config import ConfigError, Settings, load_settings
from cryptobot.data.store import DataStore

app = typer.Typer(
    help="cryptobot 暗号資産自動売買システム",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
data_app = typer.Typer(help="研究用データの取得と確認", no_args_is_help=True)
universe_app = typer.Typer(help="対象銘柄（ユニバース）の確認", no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(universe_app, name="universe")

ConfigOpt = Annotated[
    Path | None, typer.Option("--config", "-c", help="設定ファイルの場所（通常は省略）")
]


def _settings(path: Path | None) -> Settings:
    try:
        return load_settings(path)
    except ConfigError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=2) from e


def _store(s: Settings) -> DataStore:
    return DataStore(s.data.root, s.data.interval)


@app.callback()
def _main() -> None:
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_cols(12)
    pl.Config.set_fmt_str_lengths(40)


@app.command()
def version() -> None:
    """バージョンを表示。"""
    typer.echo(f"cryptobot {__version__}")


@app.command()
def doctor(config: ConfigOpt = None) -> None:
    """環境診断（ネットワーク、設定、ディスク）。"""
    from cryptobot.doctor import format_report, run_checks

    report, ok = format_report(run_checks(config))
    typer.echo(report)
    raise typer.Exit(code=0 if ok else 1)


@data_app.command("symbols")
def data_symbols(
    config: ConfigOpt = None,
    all_symbols: Annotated[bool, typer.Option("--all", help="無期限先物以外も含めて表示")] = False,
) -> None:
    """配布されている銘柄の一覧を表示。"""
    from cryptobot.data.binance_vision import filter_perp_symbols, list_symbols, make_client

    s = _settings(config)
    with make_client() as client:
        names = list_symbols(client)
    if not all_symbols:
        names = filter_perp_symbols(names, s.universe.quote)
    typer.echo("\n".join(names))
    typer.echo(f"\n合計 {len(names)} 銘柄", err=True)


@data_app.command("download")
def data_download(
    config: ConfigOpt = None,
    symbols: Annotated[
        str | None,
        typer.Option("--symbols", help="カンマ区切りで銘柄を指定（例: BTCUSDT,ETHUSDT）"),
    ] = None,
    limit: Annotated[int | None, typer.Option("--limit", help="先頭から N 銘柄だけ取得")] = None,
    start: Annotated[str | None, typer.Option("--start", help="取得開始年月 YYYY-MM")] = None,
    funding: Annotated[bool, typer.Option(help="ファンディングレートも取得する")] = True,
) -> None:
    """過去データを取得して Parquet に変換する。2 回目以降は差分だけ取得する。"""
    from cryptobot.data.binance_vision import (
        SyncResult,
        filter_perp_symbols,
        list_symbols,
        make_client,
        sync_many,
    )

    s = _settings(config)
    store = _store(s)
    start_month = start or s.data.start_month
    with make_client() as client:
        if symbols:
            names = [x.strip().upper() for x in symbols.split(",") if x.strip()]
        else:
            names = filter_perp_symbols(list_symbols(client), s.universe.quote)
        if limit is not None:
            names = names[:limit]
        typer.echo(
            f"{len(names)} 銘柄を {start_month} 以降について同期します（保存先: {store.root}）"
        )

        done = 0

        def on_result(r: SyncResult) -> None:
            nonlocal done
            done += 1
            if r.error:
                typer.echo(f"[{done}/{len(names)}] {r.symbol}: 失敗 {r.error}")
                return
            span = f"{r.first:%Y-%m-%d}〜{r.last:%Y-%m-%d}" if r.first and r.last else "データなし"
            extra = f"、注意: {'; '.join(r.warnings)}" if r.warnings else ""
            typer.echo(
                f"[{done}/{len(names)}] {r.symbol}: 新規 {r.files_downloaded} ファイル、"
                f"{r.kline_rows} 本 ({span}){extra}"
            )

        results = sync_many(
            client,
            store,
            names,
            s.data.interval,
            start_month,
            workers=s.data.parallel_downloads,
            verify_checksum=s.data.verify_checksum,
            include_funding=funding,
            on_result=on_result,
        )
    failed = [r for r in results if r.error]
    typer.echo(f"\n完了: 成功 {len(results) - len(failed)}、失敗 {len(failed)}")
    if failed:
        typer.echo("失敗した銘柄: " + ", ".join(r.symbol for r in failed))
        raise typer.Exit(code=1)


@data_app.command("status")
def data_status(config: ConfigOpt = None) -> None:
    """取得済みデータの一覧（銘柄、期間、本数）。"""
    s = _settings(config)
    store = _store(s)
    df = store.summary()
    if df.is_empty():
        typer.echo("まだデータがありません。`make data` で取得してください。")
        return
    typer.echo(str(df))
    typer.echo(f"\n{df.height} 銘柄、合計 {int(df['rows'].sum()):,} 本")


@universe_app.command("show")
def universe_show(
    config: ConfigOpt = None,
    as_of: Annotated[
        str | None, typer.Option("--as-of", help="この日付時点で選ぶ（YYYY-MM-DD、省略時は今）")
    ] = None,
) -> None:
    """設定に基づいて対象銘柄を選び、表示する。"""
    from cryptobot.data.universe import latest_bar_time, select_universe

    s = _settings(config)
    store = _store(s)
    if as_of:
        when = datetime.fromisoformat(as_of).replace(tzinfo=UTC)
    else:
        latest = latest_bar_time(store)
        if latest is None:
            typer.echo("まだデータがありません。`make data` で取得してください。")
            raise typer.Exit(code=1)
        when = latest + timedelta(hours=1)
    df = select_universe(store, when, s.universe)
    if df.is_empty():
        typer.echo("条件を満たす銘柄がありません。データが取得済みか確認してください。")
        raise typer.Exit(code=1)
    typer.echo(f"{when:%Y-%m-%d %H:%M} UTC 時点、直近 {s.universe.lookback_days} 日の売買代金上位")
    typer.echo(
        str(
            df.with_columns(
                (pl.col("dollar_volume") / 1e6).round(1).alias("dollar_volume_musd")
            ).drop("dollar_volume")
        )
    )


if __name__ == "__main__":
    app()
