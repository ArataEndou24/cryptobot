"""ベクトル化バックテスト。

前提と簡略化（正直に書く）:
- 目標ウェイト W[t] は時刻 t の足の終値までの情報で決め、
  次の足 t+1 の値動きに適用する（1 本の遅延）。
- ウェイトは「資産に対する割合」。足ごとに資産に比例して保有量が調整されるとみなす。
- 取引コストはウェイト変化量（回転率）× 片道コスト率。
  リバランス時以外にも W が変われば課金される。
- ファンディングはその時刻に適用されるレートに対し、ロングが支払い、ショートが受け取る。
- 価格が欠けている足では、その銘柄の損益を 0 とする（保有していれば警告を出す）。
- 清算、証拠金不足、最小注文単位はここでは扱わない。ポートフォリオ層と執行層の責務。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from cryptobot.research.panel import HOURS_PER_YEAR, Panel


@dataclass
class CostModel:
    fee_bps: float = 4.5
    slippage_bps: float = 5.0

    @property
    def one_way_rate(self) -> float:
        return (self.fee_bps + self.slippage_bps) / 10_000.0


@dataclass
class BacktestResult:
    times: np.ndarray
    net_returns: np.ndarray  # 足ごとの純リターン（資産比）
    gross_returns: np.ndarray
    funding_returns: np.ndarray
    cost_returns: np.ndarray  # 負の値
    turnover: np.ndarray  # 足ごとの回転率（ウェイト変化の合計）
    gross_leverage: np.ndarray  # 足ごとの総建玉（|W| の合計）
    equity: np.ndarray
    warnings: list[str] = field(default_factory=list)

    def stats(self) -> dict[str, float]:
        r = self.net_returns
        n = len(r)
        years = n / HOURS_PER_YEAR if n else 0.0
        total = float(self.equity[-1] / self.equity[0] - 1.0) if n else 0.0
        cagr = float((1.0 + total) ** (1.0 / years) - 1.0) if years > 0 and total > -1 else 0.0
        vol = float(np.std(r, ddof=1) * np.sqrt(HOURS_PER_YEAR)) if n > 1 else 0.0
        mean_ann = float(np.mean(r) * HOURS_PER_YEAR) if n else 0.0
        sharpe = mean_ann / vol if vol > 0 else 0.0
        peak = np.maximum.accumulate(self.equity)
        dd = self.equity / peak - 1.0
        max_dd = float(dd.min()) if n else 0.0
        daily = _daily_returns(self.times, r)
        return {
            "years": years,
            "total_return": total,
            "cagr": cagr,
            "annual_vol": vol,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "calmar": (cagr / -max_dd) if max_dd < 0 else 0.0,
            "avg_gross_leverage": float(np.mean(self.gross_leverage)) if n else 0.0,
            "annual_turnover": float(np.sum(self.turnover) / years) if years > 0 else 0.0,
            "annual_cost": float(np.sum(self.cost_returns) / years) if years > 0 else 0.0,
            "annual_funding": float(np.sum(self.funding_returns) / years) if years > 0 else 0.0,
            "annual_gross": float(np.sum(self.gross_returns) / years) if years > 0 else 0.0,
            "worst_day": float(daily.min()) if daily.size else 0.0,
            "best_day": float(daily.max()) if daily.size else 0.0,
            "daily_hit_rate": float(np.mean(daily > 0)) if daily.size else 0.0,
        }

    def yearly(self) -> pl.DataFrame:
        df = pl.DataFrame({"time": self.times, "r": self.net_returns, "eq": self.equity})
        df = df.with_columns(pl.col("time").dt.year().alias("year"))
        out = df.group_by("year", maintain_order=True).agg(
            ((pl.col("r") + 1.0).product() - 1.0).alias("return"),
            (pl.col("r").std() * np.sqrt(HOURS_PER_YEAR)).alias("vol"),
            (pl.col("r").mean() * HOURS_PER_YEAR).alias("mean_ann"),
            (pl.col("eq") / pl.col("eq").cum_max() - 1.0).min().alias("max_dd_within_year"),
            pl.len().alias("bars"),
        )
        return out.with_columns(
            pl.when(pl.col("vol") > 0)
            .then(pl.col("mean_ann") / pl.col("vol"))
            .otherwise(0.0)
            .alias("sharpe")
        ).drop("mean_ann")


def _daily_returns(times: np.ndarray, r: np.ndarray) -> np.ndarray:
    if len(r) == 0:
        return np.array([])
    df = pl.DataFrame({"time": times, "r": r}).with_columns(pl.col("time").dt.date().alias("d"))
    return df.group_by("d").agg(((pl.col("r") + 1.0).product() - 1.0).alias("r"))["r"].to_numpy()


def simulate(
    panel: Panel,
    weights: np.ndarray,
    cost: CostModel,
    initial_equity: float = 1.0,
    start_index: int = 0,
    end_index: int | None = None,
) -> BacktestResult:
    """weights[t] は t の終値で決めた目標ウェイト。[start_index, end_index) を集計する。"""
    if weights.shape != panel.close.shape:
        raise ValueError("weights の形がパネルと一致しません")
    close = panel.close
    T = close.shape[0]
    ret = np.full_like(close, np.nan)
    ret[1:] = close[1:] / close[:-1] - 1.0
    w_prev = np.vstack([np.zeros((1, close.shape[1])), weights[:-1]])  # 足 t の間に保有するウェイト
    w_prev = np.nan_to_num(w_prev, nan=0.0)
    w_now = np.nan_to_num(weights, nan=0.0)

    held_missing = (np.abs(w_prev) > 0) & np.isnan(ret)
    warnings: list[str] = []
    n_missing = int(held_missing[start_index:].sum())
    if n_missing:
        warnings.append(
            f"保有中に価格が欠けていた箇所が {n_missing} 件あり、その損益を 0 としました"
        )

    gross = np.nansum(w_prev * np.nan_to_num(ret, nan=0.0), axis=1)
    funding = -np.sum(w_prev * panel.funding, axis=1)
    turnover = np.sum(np.abs(w_now - w_prev), axis=1)
    costs = -turnover * cost.one_way_rate
    net = gross + funding + costs
    lev = np.sum(np.abs(w_prev), axis=1)

    sl = slice(start_index, T if end_index is None else min(end_index, T))
    equity = initial_equity * np.cumprod(1.0 + net[sl])
    return BacktestResult(
        times=panel.times[sl],
        net_returns=net[sl],
        gross_returns=gross[sl],
        funding_returns=funding[sl],
        cost_returns=costs[sl],
        turnover=turnover[sl],
        gross_leverage=lev[sl],
        equity=equity,
        warnings=warnings,
    )
