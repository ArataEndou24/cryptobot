"""運用者向けコマンド。`cryptobot --help` で一覧が出る。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated

import numpy as np
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
backtest_app = typer.Typer(help="過去データでの検証（バックテスト）", no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(universe_app, name="universe")
app.add_typer(backtest_app, name="backtest")

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
        str | None,
        typer.Option("--as-of", help="この日付時点で選ぶ（YYYY-MM-DD、省略時はデータの最終時刻）"),
    ] = None,
) -> None:
    """設定に基づいて対象銘柄を選び、表示する。除外した非暗号資産も併せて表示する。"""
    from cryptobot.data.universe import (
        compute_daily,
        excluded_non_crypto,
        latest_bar_time,
        select_universe_from_daily,
    )

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
    daily = compute_daily(store)
    df = select_universe_from_daily(daily, when, s.universe)
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
    if s.universe.exclude_non_crypto:
        ex = excluded_non_crypto(daily, when, s.universe)
        typer.echo(f"\n非暗号資産として除外した銘柄: {ex.height} 件（週末出来高比が小さい順）")
        typer.echo(str(ex.head(40)))
        if ex.height > 40:
            typer.echo(f"... 他 {ex.height - 40} 件")


def _parse_day(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


@backtest_app.command("run")
def backtest_run(
    config: ConfigOpt = None,
    start: Annotated[str | None, typer.Option("--start", help="開始日 YYYY-MM-DD")] = None,
    end: Annotated[str | None, typer.Option("--end", help="終了日 YYYY-MM-DD")] = None,
) -> None:
    """設定ファイルの戦略を過去データで検証し、成績を表示する。"""
    from cryptobot.research.backtest import CostModel, simulate
    from cryptobot.research.panel import build_panel
    from cryptobot.research.report import format_report, format_yearly, interpret
    from cryptobot.strategy.momentum import target_weights

    s = _settings(config)
    store = _store(s)
    t_start = _parse_day(start or s.backtest.start)
    end_text = end or s.backtest.end
    t_end = _parse_day(end_text) if end_text else _latest_or_exit(store)
    warmup = max(max(s.strategy.horizons_hours), s.strategy.vol_lookback_hours) + 1
    typer.echo(f"パネルを構築中（{t_start:%Y-%m-%d} 〜 {t_end:%Y-%m-%d}、助走 {warmup} 時間）...")
    panel = build_panel(store, t_start, t_end, s.universe, warmup_hours=warmup)
    typer.echo(f"銘柄 {panel.n_symbols}、足 {panel.n_times:,} 本。目標ウェイトを計算中...")
    w = target_weights(panel, s.strategy, s.risk)
    cost = CostModel(s.backtest.fee_bps, s.backtest.slippage_bps)
    res = simulate(panel, w, cost, s.backtest.initial_equity, start_index=panel.index_of(t_start))
    typer.echo(format_report(f"検証結果: {s.strategy.name}", res))
    typer.echo(format_yearly(res))
    typer.echo("")
    typer.echo("読み方:")
    for note in interpret(res):
        typer.echo(f"  - {note}")
    no_cost = simulate(panel, w, CostModel(0, 0), 1.0, start_index=panel.index_of(t_start))
    typer.echo(
        f"  - 参考: コストをゼロにした場合のシャープレシオは {no_cost.stats()['sharpe']:.2f}"
        f"（現在 {res.stats()['sharpe']:.2f}）。差が大きいほどコストに弱い戦略です。"
    )


@backtest_app.command("walkforward")
def backtest_walkforward(
    config: ConfigOpt = None,
    start: Annotated[str | None, typer.Option("--start", help="開始日 YYYY-MM-DD")] = None,
    end: Annotated[str | None, typer.Option("--end", help="終了日 YYYY-MM-DD")] = None,
    train_days: Annotated[int, typer.Option(help="学習期間の日数")] = 365,
    test_days: Annotated[int, typer.Option(help="検証期間の日数")] = 90,
) -> None:
    """学習期間で最良の設定を選び、直後の検証期間に適用する、を繰り返す（過学習の検出）。"""
    from cryptobot.research.backtest import CostModel
    from cryptobot.research.panel import Panel, build_panel
    from cryptobot.research.report import format_report, interpret
    from cryptobot.research.walkforward import walk_forward
    from cryptobot.strategy.momentum import target_weights

    s = _settings(config)
    store = _store(s)
    t_start = _parse_day(start or s.backtest.start)
    end_text = end or s.backtest.end
    t_end = _parse_day(end_text) if end_text else _latest_or_exit(store)
    warmup = max(max(s.strategy.horizons_hours), s.strategy.vol_lookback_hours) + 1
    grid: list[dict[str, object]] = [
        {"horizons_hours": h, "trade_band": band, "rebalance_hours": rb}
        for h in ([336], [720], [168, 336, 720])
        for band in (0.03, 0.06)
        for rb in (4, 24)
    ]
    typer.echo(f"パネルを構築中（{t_start:%Y-%m-%d} 〜 {t_end:%Y-%m-%d}）...")
    panel = build_panel(store, t_start, t_end, s.universe, warmup_hours=warmup)
    typer.echo(f"銘柄 {panel.n_symbols}、足 {panel.n_times:,} 本。設定 {len(grid)} 通りで検証中...")

    def weight_fn(p: Panel, params: dict[str, object]) -> np.ndarray:
        cfg = s.strategy.model_copy(update=params)
        return target_weights(p, cfg, s.risk)

    cost = CostModel(s.backtest.fee_bps, s.backtest.slippage_bps)
    folds, combined = walk_forward(
        panel, t_start, t_end, grid, weight_fn, cost, train_days, test_days, warmup
    )
    typer.echo("区間ごとの結果（学習期間で選んだ設定 → 検証期間の成績）")
    for f in folds:
        st = f.test_result.stats()
        typer.echo(
            f"  {f.test_start:%Y-%m-%d}〜{f.test_end:%Y-%m-%d}: 設定 {f.chosen} "
            f"学習シャープ {f.train_sharpe:.2f} → 検証シャープ {st['sharpe']:.2f}、"
            f"リターン {st['total_return'] * 100:+.1f}%"
        )
    typer.echo("")
    typer.echo(format_report("ウォークフォワード合成成績（検証期間のみ）", combined))
    for note in interpret(combined):
        typer.echo(f"  - {note}")


COMPARE_VARIANTS: dict[str, dict[str, object]] = {
    "基準（設定ファイル）": {},
    "ファンディングも選定に使う": {"funding_weight": 0.5},
    "レバレッジを4時間ごと更新": {"leverage_update_hours": 4},
    "等金額配分": {"weighting": "equal"},
    "帯を6%に": {"trade_band": 0.06},
    "日次リバランス": {"rebalance_hours": 24},
    "2週間モメンタムのみ": {"horizons_hours": [336]},
    "長い期間（30〜90日）": {"horizons_hours": [720, 1440, 2160]},
    "上下2割ずつ": {
        "long_fraction": 0.2,
        "short_fraction": 0.2,
        "long_exit_fraction": 0.4,
        "short_exit_fraction": 0.4,
    },
    "ロングのみ": {"short_fraction": 0.0, "short_exit_fraction": 0.0},
}


@backtest_app.command("compare")
def backtest_compare(
    config: ConfigOpt = None,
    start: Annotated[str | None, typer.Option("--start", help="開始日 YYYY-MM-DD")] = None,
    end: Annotated[str | None, typer.Option("--end", help="終了日 YYYY-MM-DD")] = None,
) -> None:
    """戦略の構成要素を 1 つずつ外した変種を同じ期間で比べ、どこに優位性があるかを見る。"""
    from cryptobot.research.backtest import CostModel, simulate
    from cryptobot.research.panel import build_panel
    from cryptobot.strategy.momentum import target_weights

    s = _settings(config)
    store = _store(s)
    t_start = _parse_day(start or s.backtest.start)
    end_text = end or s.backtest.end
    t_end = _parse_day(end_text) if end_text else _latest_or_exit(store)
    warmup = 2160 + 1
    typer.echo(f"パネルを構築中（{t_start:%Y-%m-%d} 〜 {t_end:%Y-%m-%d}）...")
    panel = build_panel(store, t_start, t_end, s.universe, warmup_hours=warmup)
    typer.echo(
        f"銘柄 {panel.n_symbols}、足 {panel.n_times:,} 本。{len(COMPARE_VARIANTS)} 通りを検証中..."
    )
    cost = CostModel(s.backtest.fee_bps, s.backtest.slippage_bps)
    i0 = panel.index_of(t_start)
    rows: list[str] = [
        "  変種                     年率     シャープ  最大DD   回転率   値動き   ファンディング  コスト"  # noqa: E501
    ]
    for name, params in COMPARE_VARIANTS.items():
        cfg = s.strategy.model_copy(update=params)
        res = simulate(panel, target_weights(panel, cfg, s.risk), cost, start_index=i0)
        st = res.stats()
        rows.append(
            f"  {name:<22} {st['cagr'] * 100:+7.1f}%  {st['sharpe']:6.2f}  "
            f"{st['max_drawdown'] * 100:+6.1f}%  {st['annual_turnover']:5.0f}倍  "
            f"{st['annual_gross'] * 100:+6.1f}%  {st['annual_funding'] * 100:+8.1f}%  "
            f"{st['annual_cost'] * 100:+6.1f}%"
        )
        typer.echo(rows[-1])
    typer.echo("")
    typer.echo("読み方: 年率・回転率・値動き・ファンディング・コストは全て年率換算。")
    typer.echo(
        "  「モメンタムのみ」と「ファンディングのみ」を比べると、どちらの要素が効いているかが分かります。"  # noqa: E501
    )


def _latest_or_exit(store: DataStore) -> datetime:
    from cryptobot.data.universe import latest_bar_time

    latest = latest_bar_time(store)
    if latest is None:
        typer.echo("まだデータがありません。`make data` で取得してください。")
        raise typer.Exit(code=1)
    return latest


if __name__ == "__main__":
    app()
