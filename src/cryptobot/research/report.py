"""検証結果を運用者向けの日本語で整形する。"""

from __future__ import annotations

from cryptobot.research.backtest import BacktestResult


def _pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def format_report(title: str, res: BacktestResult) -> str:
    s = res.stats()
    lines = [title, "=" * 56]
    lines.append(f"期間             : {res.times[0]} 〜 {res.times[-1]}（{s['years']:.2f} 年）")
    lines.append(f"総リターン       : {_pct(s['total_return'])}")
    lines.append(f"年率リターン     : {_pct(s['cagr'])}")
    lines.append(f"年率ボラティリティ: {_pct(s['annual_vol'])}")
    lines.append(f"シャープレシオ   : {s['sharpe']:.2f}")
    lines.append(f"最大ドローダウン : {_pct(s['max_drawdown'])}")
    lines.append(f"カルマーレシオ   : {s['calmar']:.2f}")
    lines.append(f"平均総建玉       : 資産の {s['avg_gross_leverage']:.2f} 倍")
    lines.append(f"年間回転率       : 資産の {s['annual_turnover']:.0f} 倍")
    lines.append("-" * 56)
    lines.append("年率の内訳（合計がおおよそ年率リターンに対応）")
    lines.append(f"  値動きによる損益 : {_pct(s['annual_gross'])}")
    lines.append(f"  ファンディング   : {_pct(s['annual_funding'])}")
    lines.append(f"  取引コスト       : {_pct(s['annual_cost'])}")
    lines.append("-" * 56)
    lines.append(f"最悪の 1 日      : {_pct(s['worst_day'])}")
    lines.append(f"最良の 1 日      : {_pct(s['best_day'])}")
    lines.append(f"日次勝率         : {s['daily_hit_rate'] * 100:.1f}%")
    if res.warnings:
        lines.append("-" * 56)
        lines.extend(f"注意: {w}" for w in res.warnings)
    lines.append("=" * 56)
    return "\n".join(lines)


def format_yearly(res: BacktestResult) -> str:
    df = res.yearly()
    lines = ["年ごとの成績", "  年    リターン   ボラ    シャープ  年内最大DD"]
    for row in df.iter_rows(named=True):
        lines.append(
            f"  {row['year']}  {_pct(row['return']):>8}  {_pct(row['vol']):>6}  "
            f"{row['sharpe']:>6.2f}  {_pct(row['max_dd_within_year']):>8}"
        )
    return "\n".join(lines)


def interpret(res: BacktestResult) -> list[str]:
    """数字の読み方を運用者向けに要約する。"""
    s = res.stats()
    notes: list[str] = []
    if s["years"] < 1.5:
        notes.append("検証期間が 2 年未満です。この結果だけで判断しないでください。")
    if s["sharpe"] >= 1.5:
        notes.append(
            "シャープレシオが高すぎる可能性があります。過学習やデータの問題を疑ってください。"
        )
    elif s["sharpe"] >= 0.8:
        notes.append(
            "シャープレシオは実運用の候補になる水準です。ウォークフォワードで確認してください。"
        )
    elif s["sharpe"] > 0.3:
        notes.append("シャープレシオは弱い優位性を示していますが、コストとブレで消える水準です。")
    else:
        notes.append("優位性は確認できません。この設定では運用しないでください。")
    cost_share = abs(s["annual_cost"]) / max(abs(s["annual_gross"]), 1e-9)
    if cost_share > 0.5:
        notes.append(
            "値動きの利益の半分以上を取引コストが食っています。回転率を下げる必要があります。"
        )
    if s["max_drawdown"] < -0.35:
        notes.append(
            "最大ドローダウンが停止ライン（40%）に近い水準です。目標ボラティリティを下げてください。"
        )
    return notes
