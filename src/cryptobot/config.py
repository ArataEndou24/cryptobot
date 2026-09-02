"""設定ファイル（YAML）の読み込みと検証。

設計方針:
- 未知のキーはエラーにする（運用者の打ち間違いを黙って無視しない）。
- 数値には上限・下限を付け、危険な値を設定できないようにする。
- リスク上限（risk セクション）は戦略コードから書き換えられない。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

DEFAULT_CONFIG_PATH = Path("config/settings.yaml")
EXAMPLE_CONFIG_PATH = Path("config/settings.example.yaml")
ENV_CONFIG_PATH = "CRYPTOBOT_CONFIG"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExchangeConfig(_Strict):
    name: Literal["hyperliquid"] = "hyperliquid"
    network: Literal["testnet", "mainnet"] = "testnet"


class DataConfig(_Strict):
    root: Path = Path("data")
    interval: Literal["1h"] = "1h"
    start_month: str = "2020-01"
    verify_checksum: bool = True
    parallel_downloads: int = Field(default=8, ge=1, le=32)

    @field_validator("start_month")
    @classmethod
    def _check_month(cls, v: str) -> str:
        parts = v.split("-")
        if len(parts) != 2 or not (parts[0].isdigit() and parts[1].isdigit()):
            raise ValueError('start_month は YYYY-MM 形式で書いてください（例: "2020-01"）')
        year, month = int(parts[0]), int(parts[1])
        if year < 2019 or not 1 <= month <= 12:
            raise ValueError("start_month は 2019-09 以降の年月にしてください")
        return f"{year:04d}-{month:02d}"


class UniverseConfig(_Strict):
    quote: str = "USDT"
    top_n: int = Field(default=30, ge=1, le=200)
    lookback_days: int = Field(default=30, ge=1, le=365)
    min_history_days: int = Field(default=90, ge=1, le=3650)
    exclude: list[str] = Field(default_factory=list)
    include_only: list[str] = Field(default_factory=list)
    exclude_non_crypto: bool = True
    tradable_only: bool = True


class RiskConfig(_Strict):
    target_annual_vol: float = Field(default=0.25, gt=0.0, le=1.5)
    max_gross_leverage: float = Field(default=2.0, gt=0.0, le=5.0)
    max_position_pct: float = Field(default=0.15, gt=0.0, le=1.0)
    max_daily_loss_pct: float = Field(default=0.08, gt=0.0, le=0.5)
    max_drawdown_pct: float = Field(default=0.40, gt=0.0, le=0.5)
    # ドローダウンに応じた建玉の縮小。[[しきい値, 倍率], ...]。例: 15% で半分、25% で 4 分の 1。
    drawdown_scaling: list[list[float]] = Field(default_factory=list)

    @field_validator("drawdown_scaling")
    @classmethod
    def _check_scaling(cls, v: list[list[float]]) -> list[list[float]]:
        for pair in v:
            if len(pair) != 2 or not (0 < pair[0] < 1) or not (0 <= pair[1] <= 1):
                raise ValueError(
                    "drawdown_scaling は [[しきい値(0〜1), 倍率(0〜1)], ...] の形にしてください"
                )
        return sorted(v, key=lambda p: p[0])


class NotifyConfig(_Strict):
    telegram_enabled: bool = False


class StrategyConfig(_Strict):
    """第 1 世代戦略（クロスセクショナル・モメンタム + ファンディング・キャリー）の設定。"""

    name: Literal["momentum_carry"] = "momentum_carry"
    horizons_hours: list[int] = Field(default_factory=lambda: [720, 1440, 2160])
    vol_lookback_hours: int = Field(default=720, ge=24, le=8760)
    rebalance_hours: int = Field(default=4, ge=1, le=168)
    long_fraction: float = Field(default=0.3, ge=0.0, le=0.5)
    short_fraction: float = Field(default=0.3, ge=0.0, le=0.5)
    long_exit_fraction: float = Field(default=0.5, ge=0.0, le=0.5)
    short_exit_fraction: float = Field(default=0.5, ge=0.0, le=0.5)
    trade_band: float = Field(default=0.03, ge=0.0, le=0.2)
    trade_band_mode: Literal["absolute", "relative"] = "absolute"
    momentum_weight: float = Field(default=1.0, ge=0.0, le=5.0)
    funding_weight: float = Field(default=0.0, ge=0.0, le=5.0)
    funding_lookback_hours: int = Field(default=72, ge=8, le=720)
    weighting: Literal["inverse_vol", "equal"] = "equal"
    leverage_update_hours: int = Field(default=24, ge=1, le=720)

    @model_validator(mode="after")
    def _check_exit_after_entry(self) -> StrategyConfig:
        if self.long_exit_fraction < self.long_fraction:
            raise ValueError("long_exit_fraction は long_fraction 以上にしてください")
        if self.short_exit_fraction < self.short_fraction:
            raise ValueError("short_exit_fraction は short_fraction 以上にしてください")
        return self

    @field_validator("horizons_hours")
    @classmethod
    def _check_horizons(cls, v: list[int]) -> list[int]:
        if not v or any(h < 1 or h > 8760 for h in v):
            raise ValueError("horizons_hours は 1〜8760 の整数を 1 つ以上並べてください")
        return sorted(set(v))


class BacktestConfig(_Strict):
    """検証（バックテスト）の設定。コストは意図的に保守的な既定値にしてある。"""

    start: str = "2022-01-01"
    end: str | None = None
    fee_bps: float = Field(default=4.5, ge=0.0, le=100.0)
    slippage_bps: float = Field(default=5.0, ge=0.0, le=100.0)
    initial_equity: float = Field(default=200_000.0, gt=0.0)


class Settings(_Strict):
    exchange: ExchangeConfig = ExchangeConfig()
    data: DataConfig = DataConfig()
    universe: UniverseConfig = UniverseConfig()
    risk: RiskConfig = RiskConfig()
    notify: NotifyConfig = NotifyConfig()
    strategy: StrategyConfig = StrategyConfig()
    backtest: BacktestConfig = BacktestConfig()


class ConfigError(Exception):
    """設定ファイルに問題があるときの例外。メッセージは運用者向けの日本語。"""


def resolve_config_path(explicit: Path | None = None) -> tuple[Path, bool]:
    """設定ファイルの場所を決める。

    優先順位: 明示指定 > 環境変数 CRYPTOBOT_CONFIG > config/settings.yaml > 例ファイル。
    戻り値の 2 つ目は「例ファイルにフォールバックしたか」。
    """
    if explicit is not None:
        return explicit, False
    env = os.environ.get(ENV_CONFIG_PATH)
    if env:
        return Path(env), False
    if DEFAULT_CONFIG_PATH.exists():
        return DEFAULT_CONFIG_PATH, False
    return EXAMPLE_CONFIG_PATH, True


def load_settings(path: Path | None = None) -> Settings:
    """設定を読み込んで検証する。問題があれば ConfigError を投げる。"""
    resolved, _ = resolve_config_path(path)
    if not resolved.exists():
        raise ConfigError(
            f"設定ファイルが見つかりません: {resolved}\n"
            f"  `cp {EXAMPLE_CONFIG_PATH} {DEFAULT_CONFIG_PATH}` を実行してください。"
        )
    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"設定ファイルの書式が壊れています: {resolved}\n  {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(
            f"設定ファイルの最上位は key: value の形式である必要があります: {resolved}"
        )
    try:
        return Settings.model_validate(raw)
    except ValidationError as e:
        lines = [f"設定ファイルに誤りがあります: {resolved}"]
        for err in e.errors():
            loc = ".".join(str(p) for p in err["loc"]) or "(最上位)"
            lines.append(f"  - {loc}: {err['msg']}")
        raise ConfigError("\n".join(lines)) from e
