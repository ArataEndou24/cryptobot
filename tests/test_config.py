from __future__ import annotations

from pathlib import Path

import pytest

from cryptobot.config import EXAMPLE_CONFIG_PATH, ConfigError, Settings, load_settings


def test_example_config_is_valid() -> None:
    s = load_settings(Path(EXAMPLE_CONFIG_PATH))
    assert isinstance(s, Settings)
    assert s.exchange.network == "testnet"
    assert s.risk.max_drawdown_pct <= 0.5


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    p = tmp_path / "s.yaml"
    p.write_text("risk:\n  max_drawdwon_pct: 0.3\n", encoding="utf-8")
    with pytest.raises(ConfigError) as ei:
        load_settings(p)
    assert "max_drawdwon_pct" in str(ei.value)


def test_dangerous_values_are_rejected(tmp_path: Path) -> None:
    p = tmp_path / "s.yaml"
    p.write_text("risk:\n  max_gross_leverage: 50\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(p)


def test_start_month_normalized(tmp_path: Path) -> None:
    p = tmp_path / "s.yaml"
    p.write_text('data:\n  start_month: "2021-3"\n', encoding="utf-8")
    assert load_settings(p).data.start_month == "2021-03"


def test_broken_yaml(tmp_path: Path) -> None:
    p = tmp_path / "s.yaml"
    p.write_text("risk: [\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(p)
