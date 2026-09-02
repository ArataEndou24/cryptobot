"""ウォークフォワード検証。

過去（学習期間）で最も良かった設定を、その直後（検証期間）にそのまま使う、を繰り返す。
検証期間だけをつなげた成績が「本当にその時点で得られたはずの成績」に近い。
学習期間で選んだ設定が検証期間でも通用しないなら、それは過学習の兆候である。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from cryptobot.research.backtest import BacktestResult, CostModel, simulate
from cryptobot.research.panel import Panel

WeightFn = Callable[[Panel, dict[str, object]], np.ndarray]


@dataclass
class Fold:
    train_start: datetime
    test_start: datetime
    test_end: datetime
    chosen: dict[str, object]
    train_sharpe: float
    test_result: BacktestResult


def walk_forward(
    panel: Panel,
    start: datetime,
    end: datetime,
    grid: Sequence[dict[str, object]],
    weight_fn: WeightFn,
    cost: CostModel,
    train_days: int = 365,
    test_days: int = 90,
    warmup_hours: int = 0,
) -> tuple[list[Fold], BacktestResult]:
    folds: list[Fold] = []
    test_start = start + timedelta(days=train_days)
    all_net: list[np.ndarray] = []
    all_times: list[np.ndarray] = []
    parts: dict[str, list[np.ndarray]] = {
        k: [] for k in ("gross", "funding", "cost", "turnover", "lev")
    }
    while test_start < end:
        test_end = min(test_start + timedelta(days=test_days), end)
        train_start = test_start - timedelta(days=train_days)
        best: tuple[float, dict[str, object]] | None = None
        for params in grid:
            res = _run(panel, train_start, test_start, params, weight_fn, cost, warmup_hours)
            s = res.stats()["sharpe"]
            if best is None or s > best[0]:
                best = (s, params)
        assert best is not None
        test_res = _run(panel, test_start, test_end, best[1], weight_fn, cost, warmup_hours)
        folds.append(Fold(train_start, test_start, test_end, best[1], best[0], test_res))
        all_net.append(test_res.net_returns)
        all_times.append(test_res.times)
        parts["gross"].append(test_res.gross_returns)
        parts["funding"].append(test_res.funding_returns)
        parts["cost"].append(test_res.cost_returns)
        parts["turnover"].append(test_res.turnover)
        parts["lev"].append(test_res.gross_leverage)
        test_start = test_end
    if not folds:
        raise ValueError(
            "ウォークフォワードの区間が作れません。期間を長くするか train_days を短くしてください。"
        )
    net = np.concatenate(all_net)
    combined = BacktestResult(
        times=np.concatenate(all_times),
        net_returns=net,
        gross_returns=np.concatenate(parts["gross"]),
        funding_returns=np.concatenate(parts["funding"]),
        cost_returns=np.concatenate(parts["cost"]),
        turnover=np.concatenate(parts["turnover"]),
        gross_leverage=np.concatenate(parts["lev"]),
        equity=np.cumprod(1.0 + net),
    )
    return folds, combined


def _run(
    panel: Panel,
    start: datetime,
    end: datetime,
    params: dict[str, object],
    weight_fn: WeightFn,
    cost: CostModel,
    warmup_hours: int,
) -> BacktestResult:
    i0 = max(0, panel.index_of(start) - warmup_hours)
    i_start = panel.index_of(start)
    i_end = panel.index_of(end)
    sub = panel.slice(i0, i_end)
    w = weight_fn(sub, params)
    return simulate(sub, w, cost, start_index=i_start - i0)
