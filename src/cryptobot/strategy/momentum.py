"""第 1 世代戦略: クロスセクショナル・モメンタム + ファンディング・キャリー。

入力はパネル（時刻 × 銘柄）、出力は目標ウェイト行列（時刻 × 銘柄、資産比）。
時刻 t の行は t の終値までの情報だけで決まる（未来参照なし）。

手順（各リバランス時刻で）:
1. 各銘柄の複数期間リターンをボラティリティで割って平均し、モメンタム得点にする。
2. 直近のファンディングレートの横断的 z スコアを得点から差し引く
   （支払う側を避け、受け取る側に傾ける）。
3. ユニバース内で順位づけし、上位をロング、下位をショート。
   既に持っている銘柄は、順位が「手仕舞い線」まで落ちない限り持ち続ける（入口と出口を非対称にして回転を抑える）。
4. 各側の中でボラティリティの逆数で配分し、ロング・ショートの金額を等しくする（ドル中立）。
5. 直近の共分散から推定したポートフォリオのボラティリティが目標に合うようにレバレッジを決める。
6. 1 銘柄上限と総建玉上限で切る。
7. 目標と現在の差が小さい銘柄は売買しない（売買しない帯）。手仕舞いは常に行う。
リバランス時刻の間は、直前の目標ウェイトを保持する。
"""

from __future__ import annotations

import warnings

import numpy as np

from cryptobot.config import RiskConfig, StrategyConfig
from cryptobot.research.panel import HOURS_PER_YEAR, Panel


def rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    """列ごとの後方ローリング標準偏差（NaN は無視、標本数が window/2 未満なら NaN）。"""
    T, N = x.shape
    out = np.full((T, N), np.nan)
    valid = ~np.isnan(x)
    xv = np.where(valid, x, 0.0)
    cs = np.vstack([np.zeros((1, N)), np.cumsum(xv, axis=0)])
    cs2 = np.vstack([np.zeros((1, N)), np.cumsum(xv * xv, axis=0)])
    cn = np.vstack([np.zeros((1, N)), np.cumsum(valid, axis=0)])
    for t in range(T):
        lo = max(0, t + 1 - window)
        n = cn[t + 1] - cn[lo]
        s = cs[t + 1] - cs[lo]
        s2 = cs2[t + 1] - cs2[lo]
        ok = n >= max(2, window // 2)
        mean = np.divide(s, n, out=np.zeros(N), where=n > 0)
        var = np.divide(s2, n, out=np.zeros(N), where=n > 0) - mean * mean
        out[t] = np.where(ok, np.sqrt(np.maximum(var, 0.0)), np.nan)
    return out


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    T, N = x.shape
    valid = ~np.isnan(x)
    xv = np.where(valid, x, 0.0)
    cs = np.vstack([np.zeros((1, N)), np.cumsum(xv, axis=0)])
    cn = np.vstack([np.zeros((1, N)), np.cumsum(valid, axis=0)])
    out = np.full((T, N), np.nan)
    for t in range(T):
        lo = max(0, t + 1 - window)
        n = cn[t + 1] - cn[lo]
        out[t] = np.where(n > 0, (cs[t + 1] - cs[lo]) / np.maximum(n, 1), np.nan)
    return out


def momentum_scores(panel: Panel, cfg: StrategyConfig) -> tuple[np.ndarray, np.ndarray]:
    """(得点 T×N, 時間あたりボラティリティ T×N)。"""
    close = panel.close
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.log(close)
    r1 = np.full_like(close, np.nan)
    r1[1:] = logp[1:] - logp[:-1]
    vol = rolling_std(r1, cfg.vol_lookback_hours)
    vol = np.where(vol > 0, vol, np.nan)  # 値が動かない銘柄は対象外
    score = np.zeros_like(close)
    count = np.zeros_like(close)
    with np.errstate(divide="ignore", invalid="ignore"):
        for h in cfg.horizons_hours:
            ret_h = np.full_like(close, np.nan)
            ret_h[h:] = logp[h:] - logp[:-h]
            adj = ret_h / (vol * np.sqrt(h))
            ok = np.isfinite(adj)
            score += np.where(ok, adj, 0.0)
            count += ok
    score = np.where(count > 0, cfg.momentum_weight * score / np.maximum(count, 1), np.nan)
    if cfg.momentum_weight == 0:
        # モメンタムを使わない場合も「得点あり」として扱う（順位はファンディングだけで決まる）
        score = np.where(np.isnan(vol), np.nan, 0.0)
    if cfg.funding_weight > 0:
        # 8 時間ごとのレートを時間平均し、横断的 z スコアにする
        f_mean = rolling_mean(
            np.where(panel.funding != 0, panel.funding, np.nan), cfg.funding_lookback_hours
        )
        f_mean = np.where(np.isnan(f_mean), 0.0, f_mean)
        z = _cross_sectional_z(f_mean, panel.member)
        score = score - cfg.funding_weight * z
    return score, vol


def _cross_sectional_z(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    xm = np.where(mask, x, np.nan)
    count = mask.sum(axis=1, keepdims=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mu = np.nanmean(xm, axis=1, keepdims=True)
        sd = np.nanstd(xm, axis=1, keepdims=True)
    ok = (count >= 2) & (sd > 0)
    z = np.where(ok, (x - np.nan_to_num(mu)) / np.where(ok, sd, 1.0), 0.0)
    return np.nan_to_num(z, nan=0.0)


def target_weights(panel: Panel, cfg: StrategyConfig, risk: RiskConfig) -> np.ndarray:
    score, vol = momentum_scores(panel, cfg)
    T, N = panel.close.shape
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.log(panel.close)
    r1 = np.full_like(panel.close, np.nan)
    r1[1:] = logp[1:] - logp[:-1]
    W = np.zeros((T, N))
    hourly_target = risk.target_annual_vol / np.sqrt(HOURS_PER_YEAR)
    prev = np.zeros(N)
    for t in range(T):
        if t % cfg.rebalance_hours != 0:
            W[t] = prev
            continue
        ok = panel.member[t] & ~np.isnan(score[t]) & ~np.isnan(vol[t]) & (vol[t] > 0)
        idx = np.where(ok)[0]
        if idx.size < 4:
            prev = np.zeros(N)
            W[t] = prev
            continue
        longs, shorts = _select_sides(score[t], idx, prev, cfg)
        raw = np.zeros(N)
        if longs.size:
            inv = 1.0 / vol[t, longs]
            raw[longs] = 0.5 * inv / inv.sum()
        if shorts.size:
            inv = 1.0 / vol[t, shorts]
            raw[shorts] = -0.5 * inv / inv.sum()
        if longs.size == 0 or shorts.size == 0:
            raw *= 2.0  # 片側だけなら総建玉 1 に揃える
        lev = _vol_target_leverage(r1, t, raw, cfg.vol_lookback_hours, hourly_target)
        w = raw * lev
        w = np.clip(w, -risk.max_position_pct, risk.max_position_pct)
        w = _re_neutralize(w)
        gross = np.abs(w).sum()
        if gross > risk.max_gross_leverage:
            w *= risk.max_gross_leverage / gross
        w = _apply_trade_band(w, prev, cfg.trade_band)
        gross = np.abs(w).sum()
        if gross > risk.max_gross_leverage:  # 帯で残した分を含めても上限は必ず守る
            w *= risk.max_gross_leverage / gross
        prev = w
        W[t] = w
    return W


def _select_sides(
    score_t: np.ndarray, idx: np.ndarray, prev: np.ndarray, cfg: StrategyConfig
) -> tuple[np.ndarray, np.ndarray]:
    """順位に基づきロング・ショートの銘柄を決める。保有中の銘柄には緩い手仕舞い線を使う。"""
    n = idx.size
    order = idx[np.argsort(score_t[idx])]  # 昇順
    pos = np.empty(n, dtype=float)
    pos[np.argsort(score_t[idx])] = np.arange(n)
    frac_from_bottom = (pos + 0.5) / n
    frac_from_top = 1.0 - frac_from_bottom
    held_long = prev[idx] > 0
    held_short = prev[idx] < 0
    is_long = (frac_from_top < cfg.long_fraction) | (
        held_long & (frac_from_top < cfg.long_exit_fraction)
    )
    is_short = (frac_from_bottom < cfg.short_fraction) | (
        held_short & (frac_from_bottom < cfg.short_exit_fraction)
    )
    both = is_long & is_short
    is_long &= ~both
    is_short &= ~both
    del order
    return idx[is_long], idx[is_short]


def _apply_trade_band(target: np.ndarray, prev: np.ndarray, band: float) -> np.ndarray:
    """目標との差が帯の内側なら現状維持。手仕舞い（目標 0）は常に実行する。"""
    if band <= 0:
        return target
    keep = (np.abs(target - prev) < band) & (target != 0)
    return np.where(keep, prev, target)


def _re_neutralize(w: np.ndarray) -> np.ndarray:
    """上限で切った後にロングとショートの金額を再び等しくする（両側ある場合のみ）。"""
    long_sum = w[w > 0].sum()
    short_sum = -w[w < 0].sum()
    if long_sum <= 0 or short_sum <= 0:
        return w
    target = min(long_sum, short_sum)
    out = w.copy()
    out[w > 0] *= target / long_sum
    out[w < 0] *= target / short_sum
    return out


def _vol_target_leverage(
    r1: np.ndarray, t: int, w: np.ndarray, window: int, hourly_target: float
) -> float:
    lo = max(0, t + 1 - window)
    idx = np.where(w != 0)[0]
    block = r1[lo : t + 1][:, idx]
    block = np.where(np.isnan(block), 0.0, block)
    if block.shape[0] < 24:
        return 0.0
    cov = np.cov(block, rowvar=False)
    cov = np.atleast_2d(cov)
    var = float(w[idx] @ cov @ w[idx])
    if var <= 0:
        return 0.0
    return float(hourly_target / np.sqrt(var))
