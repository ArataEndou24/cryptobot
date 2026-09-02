from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from cryptobot.config import RiskConfig, StrategyConfig
from cryptobot.research.backtest import CostModel, simulate
from cryptobot.research.panel import Panel
from cryptobot.strategy.momentum import rolling_mean, rolling_std, target_weights


def make_panel(close: np.ndarray, funding: np.ndarray | None = None) -> Panel:
    T, N = close.shape
    times = np.arange(
        np.datetime64("2024-01-01T00:00", "ms"),
        np.datetime64("2024-01-01T00:00", "ms") + np.timedelta64(T, "h"),
        np.timedelta64(1, "h"),
    ).astype("datetime64[ms]")
    return Panel(
        times=times,
        symbols=[f"S{j}" for j in range(N)],
        close=close,
        quote_volume=np.ones_like(close),
        funding=np.zeros_like(close) if funding is None else funding,
        member=~np.isnan(close),
    )


def test_zero_weights_is_flat() -> None:
    close = np.cumprod(1 + np.random.default_rng(0).normal(0, 0.01, (200, 3)), axis=0)
    p = make_panel(close)
    res = simulate(p, np.zeros_like(close), CostModel(0, 0))
    assert np.allclose(res.equity, 1.0)
    assert res.stats()["annual_turnover"] == 0.0


def test_buy_and_hold_one_asset_matches_price_without_costs() -> None:
    close = np.array([[100.0], [110.0], [99.0], [120.0]])
    p = make_panel(close)
    w = np.ones_like(close)
    res = simulate(p, w, CostModel(0, 0))
    # 足 0 で決めたウェイトは足 1 から効く: 1.1, 0.9, 1.2121...
    assert res.equity[-1] == pytest.approx(120.0 / 100.0)
    assert res.turnover[0] == 1.0 and res.turnover[1] == 0.0


def test_costs_charged_on_turnover() -> None:
    close = np.full((5, 1), 100.0)
    p = make_panel(close)
    w = np.array([[1.0], [1.0], [-1.0], [-1.0], [0.0]])
    res = simulate(p, w, CostModel(fee_bps=10, slippage_bps=0))
    # 回転: 1 (建て) + 2 (ドテン) + 1 (手仕舞い) = 4、コスト 4 × 0.1% = 0.4%
    assert res.turnover.sum() == pytest.approx(4.0)
    assert res.cost_returns.sum() == pytest.approx(-0.004)
    assert res.equity[-1] < 1.0


def test_funding_sign() -> None:
    close = np.full((4, 1), 100.0)
    funding = np.array([[0.0], [0.001], [0.0], [0.001]])
    p = make_panel(close, funding)
    long_res = simulate(p, np.ones_like(close), CostModel(0, 0))
    short_res = simulate(p, -np.ones_like(close), CostModel(0, 0))
    # ロングは支払い、ショートは受け取り
    assert long_res.funding_returns.sum() == pytest.approx(-0.002)
    assert short_res.funding_returns.sum() == pytest.approx(+0.002)


def test_missing_price_while_holding_warns() -> None:
    close = np.array([[100.0], [np.nan], [100.0]])
    p = make_panel(close)
    res = simulate(p, np.ones_like(close), CostModel(0, 0))
    assert res.warnings


def test_rolling_helpers() -> None:
    x = np.array([[1.0], [2.0], [3.0], [4.0]])
    m = rolling_mean(x, 2)
    assert m[-1, 0] == pytest.approx(3.5)
    s = rolling_std(x, 4)
    assert s[-1, 0] == pytest.approx(np.std([1, 2, 3, 4]))
    assert np.isnan(s[0, 0])


def test_target_weights_respect_caps_and_no_lookahead() -> None:
    rng = np.random.default_rng(1)
    T, N = 900, 10
    drift = np.linspace(-0.003, 0.003, N)
    rets = rng.normal(0, 0.01, (T, N)) + drift
    close = 100 * np.cumprod(1 + rets, axis=0)
    p = make_panel(close)
    scfg = StrategyConfig(
        horizons_hours=[24, 72],
        vol_lookback_hours=120,
        rebalance_hours=4,
        funding_weight=0.0,
        trade_band=0.0,
    )
    risk = RiskConfig(max_position_pct=0.2, max_gross_leverage=1.5)
    W = target_weights(p, scfg, risk)
    assert np.all(np.abs(W) <= 0.2 + 1e-12)
    assert np.all(np.abs(W).sum(axis=1) <= 1.5 + 1e-9)
    # ドル中立: ロングとショートの合計はほぼ 0
    active = np.abs(W).sum(axis=1) > 0
    assert active.any()
    assert np.allclose(W[active].sum(axis=1), 0.0, atol=1e-9)
    # 上昇トレンドの銘柄はロング側、下落トレンドの銘柄はショート側に偏る
    assert W[-200:, -1].mean() > 0 and W[-200:, 0].mean() < 0
    # 未来参照なし: 後半の価格を変えても前半のウェイトは変わらない
    close2 = close.copy()
    close2[600:] *= rng.uniform(0.5, 1.5, (T - 600, N))
    W2 = target_weights(make_panel(close2), scfg, risk)
    assert np.array_equal(W[:600], W2[:600])


def test_panel_index_of() -> None:
    p = make_panel(np.ones((10, 1)))
    assert p.index_of(datetime(2024, 1, 1, 3, tzinfo=UTC)) == 3


def test_walk_forward_runs_and_concatenates_test_periods() -> None:
    from datetime import timedelta

    from cryptobot.research.walkforward import walk_forward

    rng = np.random.default_rng(3)
    T, N = 24 * 400, 8
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, (T, N)), axis=0)
    p = make_panel(close)
    risk = RiskConfig()
    grid: list[dict[str, object]] = [{"horizons_hours": [24]}, {"horizons_hours": [72]}]

    def weight_fn(panel: Panel, params: dict[str, object]) -> np.ndarray:
        cfg = StrategyConfig(vol_lookback_hours=120, funding_weight=0.0).model_copy(update=params)
        return target_weights(panel, cfg, risk)

    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(days=400)
    folds, combined = walk_forward(
        p, start, end, grid, weight_fn, CostModel(0, 0), train_days=100, test_days=50
    )
    assert len(folds) == 6
    assert combined.net_returns.shape[0] == 300 * 24
    assert all(f.chosen in grid for f in folds)


def test_leverage_update_interval_and_equal_weighting() -> None:
    rng = np.random.default_rng(5)
    T, N = 24 * 60, 12
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, (T, N)), axis=0)
    p = make_panel(close)
    risk = RiskConfig()
    base = StrategyConfig(
        horizons_hours=[48],
        vol_lookback_hours=120,
        funding_weight=0.0,
        trade_band=0.0,
        weighting="equal",
    )
    w_fast = target_weights(p, base.model_copy(update={"leverage_update_hours": 4}), risk)
    w_slow = target_weights(p, base.model_copy(update={"leverage_update_hours": 24}), risk)

    def gross_changes(w: np.ndarray) -> int:
        g = np.abs(w).sum(axis=1)
        return int((np.abs(np.diff(g)) > 1e-12).sum())

    # 等金額配分ならレバレッジ更新か銘柄入れ替えのときだけ総建玉が変わる
    assert gross_changes(w_slow) <= gross_changes(w_fast)
    active = w_slow[-1][w_slow[-1] > 0]
    assert active.size > 0 and np.allclose(active, active[0])


def test_drawdown_scaling_reduces_exposure_after_losses() -> None:
    from cryptobot.research.backtest import apply_drawdown_scaling

    # 価格が 30% 下がり続けるとき、ロング固定のウェイトは縮められる
    close = 100 * np.cumprod(np.full((600, 1), 0.998), axis=0)
    p = make_panel(close)
    w = np.ones_like(close)
    scaled = apply_drawdown_scaling(p, w, CostModel(0, 0), [[0.15, 0.5], [0.25, 0.25]])
    assert scaled[0, 0] == 1.0
    assert scaled[-1, 0] == 0.25
    assert np.all(np.diff(scaled[:, 0]) <= 0)
    res_plain = simulate(p, w, CostModel(0, 0))
    res_scaled = simulate(p, w, CostModel(0, 0), drawdown_scaling=[[0.15, 0.5], [0.25, 0.25]])
    assert res_scaled.stats()["max_drawdown"] > res_plain.stats()["max_drawdown"]
    assert simulate(p, w, CostModel(0, 0), drawdown_scaling=[]).equity[-1] == res_plain.equity[-1]
