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
exchange_app = typer.Typer(help="取引所（Hyperliquid）の情報", no_args_is_help=True)
live_app = typer.Typer(help="運用（テストネット / 本番）", no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(universe_app, name="universe")
app.add_typer(backtest_app, name="backtest")
app.add_typer(exchange_app, name="exchange")
app.add_typer(live_app, name="live")

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
    _load_tradable(s)
    return DataStore(s.data.root, s.data.interval)


def _load_tradable(s: Settings) -> None:
    """取引所の上場一覧をキャッシュから読み、ユニバースの絞り込みに使う。"""
    from cryptobot.data.universe import set_tradable
    from cryptobot.exchange.hyperliquid_meta import load_markets, map_symbols

    if not s.universe.tradable_only:
        set_tradable(None)
        return
    markets, fetched_at = load_markets(s.data.root)
    if not markets:
        typer.echo(
            "注意: 取引所の上場一覧が未取得のため、取引できない銘柄も対象に含まれます。"
            " `make exchange-symbols` で取得してください。",
            err=True,
        )
        set_tradable(None)
        return
    store = DataStore(s.data.root, s.data.interval)
    mapping = map_symbols(store.symbols(), markets, s.universe.quote)
    set_tradable(mapping.keys())
    day = fetched_at[:10] if fetched_at else "?"
    typer.echo(f"取引所で取引可能な銘柄に限定: {len(mapping)} 銘柄（一覧取得 {day}）", err=True)


@exchange_app.command("symbols")
def exchange_symbols(config: ConfigOpt = None) -> None:
    """Hyperliquid の上場銘柄一覧を取得して保存し、研究データとの対応を表示する。"""
    from cryptobot.exchange.hyperliquid_meta import fetch_markets, map_symbols, save_markets

    s = _settings(config)
    markets = fetch_markets("mainnet")
    path = save_markets(markets, s.data.root, "mainnet")
    store = DataStore(s.data.root, s.data.interval)
    mapping = map_symbols(store.symbols(), markets, s.universe.quote)
    typer.echo(
        f"Hyperliquid 上場: {len(markets)} 銘柄、研究データと対応がついた銘柄: {len(mapping)}"
    )
    typer.echo(f"保存先: {path}")
    renamed = {b: h for b, h in mapping.items() if h != b.removesuffix(s.universe.quote)}
    if renamed:
        typer.echo("名前が異なる対応: " + ", ".join(f"{b}→{h}" for b, h in sorted(renamed.items())))


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
    res = simulate(
        panel,
        w,
        cost,
        s.backtest.initial_equity,
        start_index=panel.index_of(t_start),
        drawdown_scaling=s.risk.drawdown_scaling,
    )
    typer.echo(format_report(f"検証結果: {s.strategy.name}", res))
    typer.echo(format_yearly(res))
    typer.echo("")
    typer.echo("読み方:")
    for note in interpret(res):
        typer.echo(f"  - {note}")
    no_cost = simulate(
        panel,
        w,
        CostModel(0, 0),
        1.0,
        start_index=panel.index_of(t_start),
        drawdown_scaling=s.risk.drawdown_scaling,
    )
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
    warmup = 2160 + 1
    grid: list[dict[str, object]] = [
        {"horizons_hours": h, "rebalance_hours": rb, "weighting": wt}
        for h in ([336], [720, 1440, 2160], [336, 720, 1440])
        for rb in (4, 24)
        for wt in ("inverse_vol", "equal")
    ]
    typer.echo(f"パネルを構築中（{t_start:%Y-%m-%d} 〜 {t_end:%Y-%m-%d}）...")
    panel = build_panel(store, t_start, t_end, s.universe, warmup_hours=warmup)
    typer.echo(f"銘柄 {panel.n_symbols}、足 {panel.n_times:,} 本。設定 {len(grid)} 通りで検証中...")

    def weight_fn(p: Panel, params: dict[str, object]) -> np.ndarray:
        cfg = s.strategy.model_copy(update=params)
        return target_weights(p, cfg, s.risk)

    cost = CostModel(s.backtest.fee_bps, s.backtest.slippage_bps)
    folds, combined = walk_forward(
        panel,
        t_start,
        t_end,
        grid,
        weight_fn,
        cost,
        train_days,
        test_days,
        warmup,
        drawdown_scaling=s.risk.drawdown_scaling,
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


# 変種は戦略設定の上書き。"risk" キーがあればリスク設定も上書きする。
COMPARE_VARIANTS: dict[str, dict[str, object]] = {
    "基準（30〜90日、逆ボラ）": {},
    "2 週間、等金額": {"horizons_hours": [336], "weighting": "equal"},
    "2 週間、逆ボラ": {"horizons_hours": [336]},
    "2 週間〜90 日の 4 期間、逆ボラ": {"horizons_hours": [336, 720, 1440, 2160]},
    "2 週間〜90 日の 4 期間、等金額": {
        "horizons_hours": [336, 720, 1440, 2160],
        "weighting": "equal",
    },
    "2 週間〜60 日、等金額": {"horizons_hours": [336, 720, 1440], "weighting": "equal"},
    "30〜90 日、等金額": {"weighting": "equal"},
    "DD縮小 15%/25%": {"risk": {"drawdown_scaling": [[0.15, 0.5], [0.25, 0.25]]}},
    "DD縮小 25%/35%": {"risk": {"drawdown_scaling": [[0.25, 0.5], [0.35, 0.25]]}},
}


@backtest_app.command("compare")
def backtest_compare(
    config: ConfigOpt = None,
    start: Annotated[str | None, typer.Option("--start", help="開始日 YYYY-MM-DD")] = None,
    end: Annotated[str | None, typer.Option("--end", help="終了日 YYYY-MM-DD")] = None,
) -> None:
    """戦略の構成要素を 1 つずつ外した変種を同じ期間で比べ、どこに優位性があるかを見る。"""
    from cryptobot.research.backtest import CostModel, simulate
    from cryptobot.research.panel import Panel, build_panel
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
    rows: list[str] = [
        "  変種                         年率     シャープ  最大DD   回転率   値動き   ファンディング  コスト"  # noqa: E501
    ]
    panels: dict[str, Panel] = {"": panel}
    for name, params in COMPARE_VARIANTS.items():
        special = {"risk", "universe", "cost_multiplier"}
        strategy_params = {k: v for k, v in params.items() if k not in special}
        risk_params = params.get("risk")
        cfg = s.strategy.model_copy(update=strategy_params)
        risk = s.risk.model_copy(update=risk_params) if isinstance(risk_params, dict) else s.risk
        uni_params = params.get("universe")
        key = str(sorted(uni_params.items())) if isinstance(uni_params, dict) else ""
        if key not in panels:
            ucfg = s.universe.model_copy(update=uni_params)  # type: ignore[arg-type]
            panels[key] = build_panel(store, t_start, t_end, ucfg, warmup_hours=warmup)
        pnl_panel = panels[key]
        mult = params.get("cost_multiplier")
        c = (
            CostModel(cost.fee_bps * float(mult), cost.slippage_bps * float(mult))
            if isinstance(mult, float | int)
            else cost
        )
        res = simulate(
            pnl_panel,
            target_weights(pnl_panel, cfg, risk),
            c,
            start_index=pnl_panel.index_of(t_start),
            drawdown_scaling=risk.drawdown_scaling,
        )
        st = res.stats()
        rows.append(
            f"  {name:<26} {st['cagr'] * 100:+7.1f}%  {st['sharpe']:6.2f}  "
            f"{st['max_drawdown'] * 100:+6.1f}%  {st['annual_turnover']:5.0f}倍  "
            f"{st['annual_gross'] * 100:+6.1f}%  {st['annual_funding'] * 100:+8.1f}%  "
            f"{st['annual_cost'] * 100:+6.1f}%"
        )
        typer.echo(rows[-1])
    typer.echo("")
    typer.echo("読み方: 年率・回転率・値動き・ファンディング・コストは全て年率換算。")
    typer.echo("  基準との差が、その要素を変えた効果です。")


@live_app.command("plan")
def live_plan(
    config: ConfigOpt = None,
    equity: Annotated[
        float | None,
        typer.Option(
            "--equity", help="想定する口座残高（USD）。省略時は口座から取得、なければ設定の初期資金"
        ),
    ] = None,
) -> None:
    """今この時点でボットが持ちたいポジションを表示する（注文は出さない乾式運転）。"""
    from cryptobot.env import load_env, secret
    from cryptobot.exchange.hyperliquid_info import HyperliquidInfo, account_equity
    from cryptobot.live.planner import build_plan

    s = _settings(config)
    store = _store(s)
    load_env()
    address = secret("HL_MAIN_ADDRESS")
    # 価格データは常に本番（mainnet）から取る。テストネットの価格と銘柄一覧は本番と異なるため。
    # 口座残高だけは設定のネットワーク（testnet / mainnet）から取る。
    if equity is None and address:
        with HyperliquidInfo(s.exchange.network) as acct:
            equity = account_equity(acct.user_state(address))
        typer.echo(f"口座残高（{s.exchange.network}）: {equity:,.2f} USD")
    if equity is None or equity <= 0:
        equity = s.backtest.initial_equity
        typer.echo(f"口座残高が取れないため、設定の初期資金 {equity:,.0f} USD 相当で計算します")
    typer.echo("Hyperliquid（本番）から 1 時間足を取得中...")
    with HyperliquidInfo("mainnet") as info:
        plan = build_plan(s, store, info, equity)
    n_uni = len(plan.universe)
    typer.echo(f"\n{plan.as_of:%Y-%m-%d %H:%M} UTC 時点の目標ポジション（ユニバース {n_uni} 銘柄）")
    typer.echo(
        "  銘柄        方向   ウェイト   金額(USD)     数量          現在値      得点   FR(8h)"
    )
    for p in plan.positions:
        side = "ロング" if p.weight > 0 else "ショート"
        typer.echo(
            f"  {p.coin:<10} {side:<5} {p.weight * 100:+6.1f}%  {p.notional_usd:+10.2f}  "
            f"{p.size:+12.4f}  {p.mark_px:>12.4f}  {p.score:+6.2f}  {p.funding_8h * 100:+.4f}%"
        )
    lev = plan.gross_notional / plan.equity
    typer.echo(f"\n総建玉: {plan.gross_notional:,.2f} USD（資産の {lev:.2f} 倍）")
    if plan.skipped:
        typer.echo("対象外: " + ", ".join(plan.skipped))
    typer.echo("\nこれは表示だけで、注文は出していません。")


def _trader(s: Settings) -> object:
    """鍵を読んで取引所に接続する。鍵がなければ運用者向けの案内を出して終了。"""
    from cryptobot.env import load_env, secret
    from cryptobot.exchange.hyperliquid_trader import HyperliquidTrader

    load_env()
    key = secret("HL_API_WALLET_PRIVATE_KEY")
    address = secret("HL_MAIN_ADDRESS")
    if not key or not address:
        typer.echo(
            ".env に HL_API_WALLET_PRIVATE_KEY と HL_MAIN_ADDRESS が必要です。"
            " .env.example を参考に作成してください（手順書 02）。",
            err=True,
        )
        raise typer.Exit(code=2)
    return HyperliquidTrader(s.exchange.network, key, address)


def _guard_execute(s: Settings, execute: bool) -> None:
    if execute and s.exchange.network == "mainnet" and not s.live.armed:
        typer.echo(
            "本番（mainnet）への注文は、設定 live.armed を true にしない限り出せません。"
            " 手順書 03 の条件を満たしてから有効にしてください。",
            err=True,
        )
        raise typer.Exit(code=2)


@live_app.command("once")
def live_once(
    config: ConfigOpt = None,
    execute: Annotated[
        bool, typer.Option("--execute", help="実際に注文を送る（省略時は表示のみ）")
    ] = False,
) -> None:
    """1 回のリバランス周期を実行する。--execute がなければ注文は送らず表示だけ。"""
    from cryptobot.exchange.hyperliquid_info import HyperliquidInfo
    from cryptobot.live.runner import run_cycle

    s = _settings(config)
    store = _store(s)
    _guard_execute(s, execute)
    trader = _trader(s)
    typer.echo(f"接続先: {s.exchange.network}、モード: {'注文送信' if execute else '表示のみ'}")
    with HyperliquidInfo("mainnet") as info:
        report = run_cycle(s, store, trader, info, execute)  # type: ignore[arg-type]
    typer.echo(report.summary())


@live_app.command("loop")
def live_loop(
    config: ConfigOpt = None,
    execute: Annotated[bool, typer.Option("--execute", help="実際に注文を送る")] = False,
) -> None:
    """リバランス時刻（UTC 00:00, 04:00, …の数分後）ごとに周期を実行し続ける。Ctrl+C で停止。"""
    import time

    from cryptobot.exchange.hyperliquid_info import HyperliquidInfo
    from cryptobot.live.runner import run_cycle
    from cryptobot.live.schedule import next_run_time

    s = _settings(config)
    store = _store(s)
    _guard_execute(s, execute)
    trader = _trader(s)
    mode = "注文送信" if execute else "表示のみ"
    typer.echo(f"接続先: {s.exchange.network}、モード: {mode}。Ctrl+C で停止。")
    while True:
        nxt = next_run_time(
            datetime.now(UTC), s.strategy.rebalance_hours, s.live.rebalance_delay_minutes
        )
        wait = (nxt - datetime.now(UTC)).total_seconds()
        typer.echo(f"次の実行: {nxt:%Y-%m-%d %H:%M} UTC（{wait / 60:.0f} 分後）")
        time.sleep(max(0.0, wait))
        with HyperliquidInfo("mainnet") as info:
            report = run_cycle(s, store, trader, info, execute)  # type: ignore[arg-type]
        typer.echo(report.summary())


@live_app.command("flatten")
def live_flatten(
    config: ConfigOpt = None,
    yes: Annotated[bool, typer.Option("--yes", help="確認なしで実行")] = False,
) -> None:
    """緊急停止: 全ポジションを閉じ、以後の新規注文を止める。"""
    from cryptobot.exchange.hyperliquid_trader import HyperliquidTrader
    from cryptobot.live.executor import execute, flatten_orders
    from cryptobot.live.state import LiveState

    s = _settings(config)
    if not yes and not typer.confirm(
        f"{s.exchange.network} の全ポジションを閉じて停止します。よろしいですか?"
    ):
        raise typer.Exit(code=1)
    trader = _trader(s)
    assert isinstance(trader, HyperliquidTrader)
    trader.cancel_all()
    orders = flatten_orders(trader.positions(), trader.mids(), trader.markets(), s.live)
    results = execute(trader, orders)
    for r in results:
        typer.echo(f"  [{'OK' if r.ok else 'NG'}] {r.request.coin}: {r.message}")
    path = s.live.state_dir / "state.json"
    st = LiveState.load(path)
    st.halted = True
    st.halt_reason = "運用者による緊急停止"
    st.save(path)
    typer.echo("停止しました。再開するには `cryptobot live resume`。")


@live_app.command("resume")
def live_resume(config: ConfigOpt = None) -> None:
    """停止フラグを解除する（原因を確認してから）。"""
    from cryptobot.live.state import LiveState

    s = _settings(config)
    path = s.live.state_dir / "state.json"
    st = LiveState.load(path)
    typer.echo(f"停止理由: {st.halt_reason or 'なし'}")
    st.halted = False
    st.halt_reason = ""
    st.consecutive_errors = 0
    st.save(path)
    typer.echo("再開可能にしました。")


@live_app.command("status")
def live_status(config: ConfigOpt = None) -> None:
    """口座、ポジション、運用状態を表示する。"""
    from cryptobot.exchange.hyperliquid_trader import HyperliquidTrader
    from cryptobot.live.state import LiveState

    s = _settings(config)
    trader = _trader(s)
    assert isinstance(trader, HyperliquidTrader)
    equity = trader.equity()
    pos = trader.positions()
    mids = trader.mids()
    st = LiveState.load(s.live.state_dir / "state.json")
    typer.echo(f"接続先: {s.exchange.network}、資産: {equity:,.2f} USD")
    if st.peak_equity > 0:
        dd = 1.0 - equity / st.peak_equity
        typer.echo(f"最高値 {st.peak_equity:,.2f} USD からのドローダウン: {dd:.1%}")
    typer.echo(f"停止フラグ: {'あり（' + st.halt_reason + '）' if st.halted else 'なし'}")
    typer.echo(f"最終実行: {st.last_run_at or 'なし'}")
    if not pos:
        typer.echo("ポジション: なし")
    else:
        typer.echo("ポジション:")
        for coin, sz in sorted(pos.items(), key=lambda kv: -abs(kv[1] * mids.get(kv[0], 0))):
            typer.echo(f"  {coin:<8} {sz:+g}（約 {sz * mids.get(coin, 0):+,.0f} USD）")


def _latest_or_exit(store: DataStore) -> datetime:
    from cryptobot.data.universe import latest_bar_time

    latest = latest_bar_time(store)
    if latest is None:
        typer.echo("まだデータがありません。`make data` で取得してください。")
        raise typer.Exit(code=1)
    return latest


if __name__ == "__main__":
    app()
