"""Markdown research report (output/report/backtest_report.md) and README performance block."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import BENCH, STRAT
from .utils import abs_path

log = logging.getLogger(__name__)


def pct(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.{nd}%}" if x < 0 else f"{x:.{nd}%}"


def num(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{nd}f}"


def md_table(df: pd.DataFrame, fmt: dict | None = None, index: bool = True) -> str:
    fmt = fmt or {}
    d = df.copy()
    for c in d.columns:
        f = fmt.get(c)
        if f is not None:
            d[c] = d[c].map(f)
    return d.to_markdown(index=index)


def core_table(summary: pd.DataFrame) -> pd.DataFrame:
    b, s = summary.loc[BENCH], summary.loc[STRAT]
    rows = [
        ("Final ¥100", f"¥{b['Final_Value_of_100']:,.1f}", f"¥{s['Final_Value_of_100']:,.1f}"),
        ("CAGR", pct(b["CAGR"]), pct(s["CAGR"])),
        ("Volatility", pct(b["Annualized_Volatility"]), pct(s["Annualized_Volatility"])),
        ("Sharpe", num(b["Sharpe_Ratio"]), num(s["Sharpe_Ratio"])),
        ("Max Drawdown", pct(b["Maximum_Drawdown"]), pct(s["Maximum_Drawdown"])),
        ("Best Year", b["Best_Year"], s["Best_Year"]),
        ("Worst Year", b["Worst_Year"], s["Worst_Year"]),
    ]
    return pd.DataFrame(rows, columns=["Metric", "CSI300", "Strategy"]).set_index("Metric")


def build(cfg: dict, bt: dict, an: dict, universe_info: dict, data_info: dict) -> Path:
    rep_dir = abs_path(cfg, cfg["output"]["report_dir"])
    rep_dir.mkdir(parents=True, exist_ok=True)
    summary = an["summary"]
    b, s, n = summary.loc[BENCH], summary.loc[STRAT], summary.loc[f"{STRAT}_Net"]
    btr, stotal = summary.loc["CSI300_TR"], summary.loc[f"{STRAT}_TR"]
    lev = an["levels"]
    start, end = lev.index[0].date(), lev.index[-1].date()
    ann = an["annual"]
    cyc = an["cycles"]
    crashes = an["crashes"]
    rs = an["rs"]
    roll = an["rolling_stats"]
    costs = an["costs"]
    reb = bt["summary"]
    ratio = rs["Strategy_to_CSI300_Ratio"]
    n_years_rs = (rs.index[-1] - rs.index[0]).days / 365.25
    rs_cagr = ratio.iloc[-1] ** (1 / n_years_rs) - 1
    best_years = ann.sort_values(STRAT, ascending=False).head(3)
    worst_years = ann.sort_values(STRAT).head(3)
    best_excess = ann.sort_values("Excess", ascending=False).head(3)
    worst_excess = ann.sort_values("Excess").head(3)
    crash = crashes.iloc[0] if len(crashes) else None
    outperform_years = int((ann["Excess"] > 0).sum())
    ch = "../charts/"

    L = []
    L.append(f"# CSI300 S&P Financial Viability + SPMO Momentum Index — Backtest Report\n")
    L.append(f"*Generated {pd.Timestamp.today():%Y-%m-%d}. Backtest {start} → {end}. Both indices start at "
             f"{cfg['project']['base_level']:.0f} on {start} (Growth-of-¥100 series rebased to 100).*\n")
    # ---------------------------------------------------------------- Executive summary
    L.append("## 1. Executive Summary\n")
    L.append(md_table(core_table(summary)))
    L.append("")
    L.append(f"- ¥100 invested in the CSI 300 price index on {start} became **¥{b['Final_Value_of_100']:,.1f}** by {end}; "
             f"the same ¥100 in the strategy became **¥{s['Final_Value_of_100']:,.1f}** (gross) / **¥{n['Final_Value_of_100']:,.1f}** net of "
             f"transaction costs.")
    L.append(f"- CAGR: CSI 300 {pct(b['CAGR'])} vs strategy {pct(s['CAGR'])} (net {pct(n['CAGR'])}); annualised excess return "
             f"{pct(s['Annualized_Excess_Return'])}, tracking error {pct(s['Tracking_Error'])}, information ratio {num(s['Information_Ratio'])}.")
    L.append(f"- Risk: volatility {pct(b['Annualized_Volatility'])} vs {pct(s['Annualized_Volatility'])}; maximum drawdown "
             f"{pct(b['Maximum_Drawdown'])} vs {pct(s['Maximum_Drawdown'])}; Sharpe {num(b['Sharpe_Ratio'])} vs {num(s['Sharpe_Ratio'])}; "
             f"beta {num(s['Beta_vs_CSI300'])}, Jensen alpha {pct(s['Alpha_vs_CSI300'])} p.a.")
    L.append(f"- The strategy beat the CSI 300 in {outperform_years} of {len(ann)} calendar years; rolling 3-year excess CAGR was positive "
             f"{pct(roll['rolling_3y_cagr_excess_win_rate'],1)} of the time and rolling 5-year excess CAGR {pct(roll['rolling_5y_cagr_excess_win_rate'],1)}.")
    L.append(f"- Average one-way turnover {pct(s['Average_Annual_Turnover'],1)} per year ({pct(s['Average_Rebalance_Turnover'],1)} per semi-annual "
             f"rebalance); cost drag {pct(costs['Annual_Cost_Drag'])} p.a. under the historical A-share commission / stamp-duty schedule.")
    L.append(f"- Total-return comparison (dividends reinvested): CSI 300 TR {pct(btr['CAGR'])} CAGR vs strategy TR {pct(stotal['CAGR'])}.\n")
    L.append(f"![Growth of ¥100]({ch}growth_of_100.png)\n")
    # ---------------------------------------------------------------- Methodology
    L.append("## 2. Strategy Methodology\n")
    L.append("```\nHistorical CSI300 (point-in-time) → S&P Financial Viability → SPMO Risk-Adjusted Momentum\n"
             "→ Top 20% by Momentum Score (S&P buffer) → SPMO weighting (FMC × score, company cap) → Index → Backtest vs CSI300\n```\n")
    L.append("The model is fixed. No parameter was tuned, the 20% selection ratio was not optimised and no additional factor "
             "(ROE, ROIC, PE, PB, growth, leverage, quality) is used.\n")
    L.append("### 2.1 Parent universe — point-in-time CSI 300 membership\n")
    L.append(f"Membership on every trading day is rebuilt from {universe_info['n_events']} official CSIndex constituent-adjustment "
             f"announcements (2005-2026: {universe_info['n_regular']} regular reviews, {universe_info['n_temporary']} temporary adjustments), "
             f"walking backwards from the current official list. Every intermediate list has exactly 300 securities and the list "
             f"reconstructed for 2005-07-01 equals the official full list published by CSIndex. {universe_info['n_codes']} distinct securities "
             f"were CSI 300 members at some point. A security that is not a member on the reference date has "
             f"`Eligible = False`, `Selected = False`, `Final Weight = 0`; a holding removed from the CSI 300 between rebalances is removed "
             f"from the strategy on the same effective date (weight redistributed pro rata). The engine raises an error if any holding "
             f"is ever outside the parent index.\n")
    L.append("### 2.2 S&P Financial Viability\n")
    L.append("Following the S&P U.S. Indices Methodology, a security passes if (i) net income from continuing operations of the most "
             "recently *published* quarter is positive and (ii) the sum over the four most recent consecutive published quarters is positive. "
             "A-share mapping: 持续经营净利润 (`CONTINUED_NETPROFIT`, income-statement line introduced with the 2018 CAS format) with fallback to "
             "净利润 (`NETPROFIT`, total net profit including minority interests) for earlier periods. Chinese quarterly statements are cumulative, "
             "so single quarters are differences of consecutive year-to-date figures. Only reports whose first announcement date "
             "(`NOTICE_DATE`) is on or before the reference date are used. Announcement dates that are missing or implausible (Eastmoney's "
             "`NOTICE_DATE` for periods before 2010 is the date of the following year's report in which the period appears as a comparative, "
             "lag ≈ 13 months) are replaced by the statutory filing deadline (Q1 Apr-30, Q2 Aug-31, Q3 Oct-31, annual Apr-30 of the next year), "
             "which is never earlier than the true publication date.\n")
    L.append("### 2.3 SPMO Momentum Methodology\n")
    L.append("- Momentum return = P(t−1M)/P(t−13M) − 1 on split/rights-adjusted (ex-dividend) prices, dates aligned to A-share trading days "
             "(last trading day on/before each calendar date).\n- Volatility = standard deviation of daily price returns on traded days in the same "
             "12-month window (≥150 traded days required, as in the S&P methodology).\n- RAM = momentum return / volatility; Z-scores are computed "
             "across CSI 300 members that pass Financial Viability and have a valid RAM, winsorised at ±3.\n- Momentum Score = 1 + Z (Z > 0) or "
             "1/(1 − Z) (Z ≤ 0).\n- Target count = round(20% × eligible count); S&P buffer: top 80% of the target selected automatically, current "
             "constituents ranked within 120% of the target retained by rank, remaining slots filled by rank. Parent membership always dominates "
             "the buffer.\n- Weight = FMC × Momentum Score, normalised; company cap = min("
             f"{cfg['weighting']['cap_absolute']:.0%}, {cfg['weighting']['cap_multiple_of_parent_weight']:g} × the stock's market-cap weight in the CSI 300), "
             "excess redistributed proportionally to uncapped stocks until no cap is breached (if the caps are jointly infeasible the multiple is "
             "raised in steps of 1 — recorded per rebalance in `rebalance_summary.csv`). The 9% company cap is the S&P 500 Momentum value; the "
             "multiple follows the S&P factor-index convention (lower of the cap and 20× the underlying weight) — the S&P/BOVESPA Momentum "
             "methodology uses 3×, which is jointly infeasible with ~55 constituents; both are settable in `config.yaml`.\n"
             "- Reference dates: last trading day of February and August; "
             "implementation after the close of the third Friday of March and September.\n")
    # ---------------------------------------------------------------- Data
    L.append("## 3. Data\n")
    L.append(md_table(pd.DataFrame(data_info["sources"]).set_index("Item")))
    L.append("")
    L.append(data_info["notes"])
    # ---------------------------------------------------------------- Historical constituents
    L.append("## 4. Historical Constituents\n")
    reb_show = reb[["reference_date", "implementation_date", "n_financial_eligible", "n_momentum_eligible", "n_eligible", "n_target",
                    "n_selected", "n_buffer_retained", "n_capped", "cap_multiple_used"]].copy()
    L.append(f"{len(reb)} reference dates ({reb['reference_date'].iloc[0]} … {reb['reference_date'].iloc[-1]}). "
             f"Per-date score files: `output/scores/YYYY-MM-DD.csv`; holdings: `output/holdings/YYYY-MM-DD.csv` (implementation date).\n")
    L.append(md_table(reb_show, index=False))
    L.append("")
    # ---------------------------------------------------------------- Performance
    L.append("## 5. Performance\n")
    show_cols = ["Final_Value_of_100", "Total_Return", "CAGR", "Annualized_Volatility", "Sharpe_Ratio", "Sortino_Ratio", "Maximum_Drawdown",
                 "Calmar_Ratio", "Best_Year", "Worst_Year", "Positive_Year_Ratio", "Monthly_Win_Rate"]
    fm = {"Final_Value_of_100": lambda x: f"¥{x:,.1f}", "Total_Return": pct, "CAGR": pct, "Annualized_Volatility": pct, "Sharpe_Ratio": num,
          "Sortino_Ratio": num, "Maximum_Drawdown": pct, "Calmar_Ratio": num, "Positive_Year_Ratio": lambda x: pct(x, 0),
          "Monthly_Win_Rate": lambda x: pct(x, 1)}
    L.append(md_table(summary[show_cols].T, index=True) if False else md_table(summary[show_cols].apply(lambda col: col.map(fm.get(col.name, lambda v: v)))))
    L.append("")
    L.append("Strategy-only statistics:\n")
    rel_cols = ["Annualized_Excess_Return", "Tracking_Error", "Information_Ratio", "Beta_vs_CSI300", "Alpha_vs_CSI300",
                "Average_Annual_Turnover", "Maximum_Rebalance_Turnover", "Gross_CAGR", "Net_CAGR", "Annual_Cost_Drag"]
    rel_tab = pd.DataFrame({"Strategy (gross)": [s[c] for c in rel_cols]}, index=rel_cols)
    rel_tab["Strategy (gross)"] = [pct(v) if "Turnover" in c or "Return" in c or "CAGR" in c or "Alpha" in c or "Error" in c or "Drag" in c else num(v)
                                   for c, v in zip(rel_cols, rel_tab["Strategy (gross)"])]
    L.append(md_table(rel_tab))
    L.append("")
    L.append("### 5.1 Growth of ¥100\n")
    L.append(f"![Growth of ¥100]({ch}growth_of_100.png)\n\n![Growth of ¥100 (log)]({ch}growth_of_100_log.png)\n")
    L.append("### 5.2 Candlestick Comparison\n")
    L.append("Monthly candles (Open = first trading day's open, High/Low = period extremes, Close = last trading day's close), identical date "
             "range and aligned time axis. Strategy OHLC is computed from constituents' own OHLC with fixed index shares (see README); Volume is "
             "left blank because a synthetic index has no meaningful volume.\n")
    L.append(f"![Candlestick comparison]({ch}candlestick_comparison.png)\n\n![Candlestick comparison with MA50/MA100]({ch}candlestick_comparison_ma.png)\n\n"
             f"![Strategy monthly]({ch}strategy_candlestick_ma.png)\n\n![CSI300 monthly]({ch}csi300_candlestick_ma.png)\n\n"
             f"![Daily comparison, last 500 days]({ch}candlestick_comparison_daily.png)\n")
    L.append("### 5.3 Relative Strength\n")
    rdd = an["relative_dd"]
    L.append(f"Strategy / CSI 300 ratio: {ratio.iloc[0]:.2f} → **{ratio.iloc[-1]:.2f}** ({pct(rs_cagr)} p.a.). Largest drawdown of the ratio "
             f"{pct(rdd['max_drawdown'])} from {rdd['peak_date'].date()} to {rdd['bottom_date'].date()}"
             + (f", recovered {rdd['recovery_date'].date()}." if rdd["recovery_date"] is not pd.NaT else " (not yet recovered).") + "\n")
    L.append(f"![Relative strength]({ch}relative_strength.png)\n")
    L.append("### 5.4 Annual Returns\n")
    L.append(md_table(ann, fmt={c: pct for c in ann.columns}))
    L.append(f"\n![Annual returns]({ch}annual_returns.png)\n")
    L.append("### 5.5 Rolling Returns\n")
    rtab = pd.DataFrame({"1Y excess": [roll["rolling_1y_return_excess_win_rate"], roll["rolling_1y_return_median_excess"]],
                         "3Y excess CAGR": [roll["rolling_3y_cagr_excess_win_rate"], roll["rolling_3y_cagr_median_excess"]],
                         "5Y excess CAGR": [roll["rolling_5y_cagr_excess_win_rate"], roll["rolling_5y_cagr_median_excess"]]},
                        index=["Share of windows with positive excess", "Median excess"])
    L.append(md_table(rtab, fmt={c: lambda x: pct(x, 1) for c in rtab.columns}))
    L.append(f"\n![Rolling 1Y]({ch}rolling_1y_return.png)\n\n![Rolling 3Y]({ch}rolling_3y_cagr.png)\n\n![Rolling 5Y]({ch}rolling_5y_cagr.png)\n\n"
             f"![Rolling excess]({ch}rolling_excess_return.png)\n")
    L.append("### 5.6 Drawdowns\n")
    dd = an["drawdowns"].copy()
    L.append(md_table(dd))
    L.append(f"\n![Drawdowns]({ch}drawdown_comparison.png)\n")
    # ---------------------------------------------------------------- Risk stats / turnover / costs
    L.append("## 6. Risk Statistics\n")
    L.append(f"See the table in Section 5 and `output/performance/summary.csv`. Beta vs CSI 300 {num(s['Beta_vs_CSI300'])}, tracking error "
             f"{pct(s['Tracking_Error'])}, information ratio {num(s['Information_Ratio'])}, Sortino {num(s['Sortino_Ratio'])} vs {num(b['Sortino_Ratio'])}, "
             f"Calmar {num(s['Calmar_Ratio'])} vs {num(b['Calmar_Ratio'])}.\n")
    L.append("## 7. Turnover\n")
    L.append(f"Average one-way turnover per rebalance {pct(s['Average_Rebalance_Turnover'],1)}, per year {pct(s['Average_Annual_Turnover'],1)}, "
             f"maximum at a single rebalance {pct(s['Maximum_Rebalance_Turnover'],1)}.\n")
    yt = an["yearly_turnover"].to_frame("One-way turnover")
    yt.index = yt.index.astype(str)
    L.append(md_table(yt, fmt={"One-way turnover": lambda x: pct(x, 1)}))
    L.append(f"\n![Turnover]({ch}turnover.png)\n")
    L.append("## 8. Transaction Costs\n")
    L.append("Costs are charged on one-way turnover at each implementation date using the historical A-share regime: brokerage commission "
             "(0.08% per side before 2013, 0.03% after), stamp duty (0.1% both sides → 0.3% both sides 2007-05-30 → 0.1% both sides 2008-04-24 → "
             "0.1% sell-only 2008-09-19 → 0.05% sell-only 2023-08-28), transfer fee 0.001% and 0.10% slippage per side.\n")
    L.append(md_table(pd.DataFrame({"Value": [pct(costs["Gross_CAGR"]), pct(costs["Net_CAGR"]), pct(costs["Annual_Cost_Drag"]),
                                              pct(costs["Total_Cost_Paid_fraction_of_NAV"])]},
                                   index=["Gross CAGR", "Net CAGR", "Annual cost drag", "Cumulative cost (sum of rebalance costs, % of NAV)"])))
    L.append("")
    # ---------------------------------------------------------------- Cycles / crashes
    L.append("## 9. Market Cycle Analysis\n")
    cfmt = {c: pct for c in cyc.columns if c not in ("Cycle", "Start", "End", "CSI300_Sharpe", "Strategy_Sharpe")}
    cfmt.update({"CSI300_Sharpe": num, "Strategy_Sharpe": num})
    L.append(md_table(cyc, fmt=cfmt, index=False))
    best_cycle = cyc.loc[cyc["Excess_Return"].idxmax()]
    worst_cycle = cyc.loc[cyc["Excess_Return"].idxmin()]
    L.append(f"\nThe strategy added most value in **{best_cycle['Cycle']}** (excess {pct(best_cycle['Excess_Return'])}) and least in "
             f"**{worst_cycle['Cycle']}** (excess {pct(worst_cycle['Excess_Return'])}).\n")
    L.append("## 10. Momentum Crash Analysis\n")
    L.append("Largest 3-month (63 trading day) underperformance episodes of the strategy relative to the CSI 300:\n")
    L.append(md_table(crashes, fmt={"Strategy_3M_Return": pct, "CSI300_3M_Return": pct, "Relative_3M": pct}, index=False))
    if crash is not None:
        L.append(f"\nThe largest momentum crash ended {crash['End_Date']}: strategy {pct(crash['Strategy_3M_Return'])} vs CSI 300 "
                 f"{pct(crash['CSI300_3M_Return'])} over the preceding three months ({pct(crash['Relative_3M'])} relative).\n")
    # ---------------------------------------------------------------- Conclusion
    L.append("## 11. Conclusion — answers to the 17 questions\n")
    q = [
        f"**Backtest period:** {start} → {end} (first reference date {reb['reference_date'].iloc[0]}, first implementation {start}).",
        f"**¥100 in CSI 300 →** ¥{b['Final_Value_of_100']:,.1f} (price index; ¥{btr['Final_Value_of_100']:,.1f} with dividends).",
        f"**¥100 in the strategy →** ¥{s['Final_Value_of_100']:,.1f} gross, ¥{n['Final_Value_of_100']:,.1f} net of costs (¥{stotal['Final_Value_of_100']:,.1f} total return, gross).",
        f"**CAGR:** CSI 300 {pct(b['CAGR'])}; strategy {pct(s['CAGR'])}.",
        f"**Annualised excess return:** {pct(s['Annualized_Excess_Return'])} (Jensen alpha {pct(s['Alpha_vs_CSI300'])}, beta {num(s['Beta_vs_CSI300'])}).",
        f"**Maximum drawdown:** CSI 300 {pct(b['Maximum_Drawdown'])} ({b['MDD_Peak_Date']} → {b['MDD_Bottom_Date']}); strategy {pct(s['Maximum_Drawdown'])} ({s['MDD_Peak_Date']} → {s['MDD_Bottom_Date']}).",
        f"**Sharpe:** CSI 300 {num(b['Sharpe_Ratio'])}; strategy {num(s['Sharpe_Ratio'])}.",
        f"**Annualised volatility:** CSI 300 {pct(b['Annualized_Volatility'])}; strategy {pct(s['Annualized_Volatility'])}.",
        f"**Average turnover:** {pct(s['Average_Rebalance_Turnover'],1)} one-way per rebalance, {pct(s['Average_Annual_Turnover'],1)} per year (max {pct(s['Maximum_Rebalance_Turnover'],1)}).",
        f"**Net CAGR after costs:** {pct(n['CAGR'])} (cost drag {pct(costs['Annual_Cost_Drag'])} p.a.).",
        "**Strongest strategy years:** " + ", ".join(f"{i} ({pct(v)})" for i, v in best_years[STRAT].items())
        + "; largest excess: " + ", ".join(f"{i} ({pct(v)})" for i, v in best_excess["Excess"].items()) + ".",
        "**Weakest strategy years:** " + ", ".join(f"{i} ({pct(v)})" for i, v in worst_years[STRAT].items())
        + "; worst excess: " + ", ".join(f"{i} ({pct(v)})" for i, v in worst_excess["Excess"].items()) + ".",
        (f"**Largest momentum crash:** 3 months to {crash['End_Date']} — strategy {pct(crash['Strategy_3M_Return'])} vs CSI 300 "
         f"{pct(crash['CSI300_3M_Return'])} ({pct(crash['Relative_3M'])} relative)." if crash is not None else "**Largest momentum crash:** n/a"),
        f"**Long-run outperformance:** the strategy beat the CSI 300 in {outperform_years}/{len(ann)} calendar years; the relative-strength ratio "
        f"went from 1.00 to {ratio.iloc[-1]:.2f}. " + ("Yes — outperformance is persistent." if roll['rolling_5y_cagr_excess_win_rate'] > 0.75 else
                                                        "Outperformance is positive on average but not uniform across sub-periods."),
        f"**Rolling excess win rates:** 3-year {pct(roll['rolling_3y_cagr_excess_win_rate'],1)}, 5-year {pct(roll['rolling_5y_cagr_excess_win_rate'],1)} "
        f"(1-year {pct(roll['rolling_1y_return_excess_win_rate'],1)}).",
        f"**Relative strength trend:** {'rising' if ratio.iloc[-1] > 1 else 'falling'} — {pct(rs_cagr)} p.a.; largest relative drawdown {pct(rdd['max_drawdown'])}.",
        f"**Does Financial Viability + SPMO improve the CSI 300?** CAGR {pct(b['CAGR'])} → {pct(s['CAGR'])} (net {pct(n['CAGR'])}), Sharpe "
        f"{num(b['Sharpe_Ratio'])} → {num(s['Sharpe_Ratio'])}, Sortino {num(b['Sortino_Ratio'])} → {num(s['Sortino_Ratio'])}, max drawdown "
        f"{pct(b['Maximum_Drawdown'])} → {pct(s['Maximum_Drawdown'])}, information ratio {num(s['Information_Ratio'])}. "
        + ("The improvement in long-run compounding and risk-adjusted return is material." if (s["CAGR"] > b["CAGR"] and s["Sharpe_Ratio"] > b["Sharpe_Ratio"])
           else "The evidence for an improvement is mixed."),
    ]
    L.extend(f"{i}. {t}" for i, t in enumerate(q, 1))
    L.append("")
    L.append("### Caveats\n")
    L.append(data_info["caveats"])
    text = "\n".join(L)
    path = rep_dir / "backtest_report.md"
    path.write_text(text, encoding="utf-8")
    return path
