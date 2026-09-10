"""Post-backtest analysis: performance CSVs, statistics and all charts."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from . import plots as P
from .utils import abs_path

log = logging.getLogger(__name__)

STRAT = "CSI300_SP_FV_SPMO"
BENCH = "CSI300"


def run(cfg: dict, bt: dict) -> dict:
    perf = abs_path(cfg, cfg["output"]["performance_dir"])
    charts = abs_path(cfg, cfg["output"]["charts_dir"])
    perf.mkdir(parents=True, exist_ok=True)
    charts.mkdir(parents=True, exist_ok=True)
    acfg = cfg["analysis"]
    periods = int(acfg.get("trading_days_per_year", 244))
    rf = float(acfg.get("risk_free_rate", 0.0))
    growth_base = float(cfg["project"]["growth_base"])

    s_lev = bt["strategy_pr"]["levels"]
    s_tr = bt["strategy_tr"]["levels"]
    bench = bt["benchmark"]
    bench_tr = bt["benchmark_tr"]
    common = s_lev.index.intersection(bench.index)
    s_lev, bench = s_lev.loc[common], bench.loc[common]
    s_tr = s_tr.loc[common]
    bench_tr = bench_tr.reindex(common).ffill()

    # --- index levels (base 100 as required for index_levels.csv) -------------------
    scale = growth_base / float(cfg["project"]["base_level"])
    levels = pd.DataFrame({BENCH: bench["Close"] * scale, STRAT: s_lev["Close"] * scale,
                           f"{STRAT}_Net": s_lev["Net_Close"] * scale,
                           f"{BENCH}_TR": bench_tr["Close"] * scale, f"{STRAT}_TR": s_tr["Close"] * scale,
                           f"{STRAT}_TR_Net": s_tr["Net_Close"] * scale})
    levels.index.name = "Date"
    levels.to_csv(perf / "index_levels.csv", float_format="%.6f")
    # base-1000 versions (section 9)
    pd.DataFrame({BENCH: bench["Close"], STRAT: s_lev["Close"]}).rename_axis("Date").to_csv(perf / "index_levels_base1000.csv", float_format="%.6f")

    # --- OHLC ----------------------------------------------------------------------
    s_ohlc = s_lev[["Open", "High", "Low", "Close"]].copy()
    s_ohlc["Volume"] = np.nan          # no meaningful volume for a synthetic index -> left blank
    s_ohlc.to_csv(perf / "strategy_ohlc.csv", float_format="%.4f")
    b_ohlc = bench[["Open", "High", "Low", "Close"]].copy()
    b_ohlc.to_csv(perf / "csi300_ohlc.csv", float_format="%.4f")
    ohlc = {"D": (s_ohlc, b_ohlc)}
    for f in ("W", "M"):
        so, bo = P.resample_ohlc(s_ohlc[["Open", "High", "Low", "Close"]], f), P.resample_ohlc(b_ohlc, f)
        so.to_csv(perf / f"strategy_ohlc_{f.lower()}.csv", float_format="%.4f")
        bo.to_csv(perf / f"csi300_ohlc_{f.lower()}.csv", float_format="%.4f")
        ohlc[f] = (so, bo)

    # --- relative strength --------------------------------------------------------
    rs = pd.DataFrame({"Strategy": levels[STRAT], "CSI300": levels[BENCH]})
    rs["Strategy_to_CSI300_Ratio"] = rs["Strategy"] / rs["CSI300"]
    rs.to_csv(perf / "relative_strength.csv", float_format="%.6f")

    # --- summary statistics -------------------------------------------------------
    turnover = bt["strategy_pr"]["turnover"]
    turnover.to_csv(perf / "turnover.csv", index=False, float_format="%.6f")
    sum_b = M.summary(levels[BENCH], "CSI300", rf, periods)
    sum_s = M.summary(levels[STRAT], STRAT, rf, periods)
    sum_n = M.summary(levels[f"{STRAT}_Net"], f"{STRAT}_Net", rf, periods)
    sum_btr = M.summary(levels[f"{BENCH}_TR"], "CSI300_TR", rf, periods)
    sum_str = M.summary(levels[f"{STRAT}_TR"], f"{STRAT}_TR", rf, periods)
    rel = M.relative_summary(levels[STRAT], levels[BENCH], turnover, periods)
    rel_net = M.relative_summary(levels[f"{STRAT}_Net"], levels[BENCH], None, periods)
    yearly_turnover = rel.pop("Yearly_Turnover")
    rel_net.pop("Yearly_Turnover", None)
    summary = pd.DataFrame([sum_b, sum_s, sum_n, sum_btr, sum_str]).set_index("Index")
    for k, v in rel.items():
        summary.loc[STRAT, k] = v
    for k, v in rel_net.items():
        summary.loc[f"{STRAT}_Net", k] = v
    costs = {"Gross_CAGR": sum_s["CAGR"], "Net_CAGR": sum_n["CAGR"], "Annual_Cost_Drag": sum_s["CAGR"] - sum_n["CAGR"],
             "Total_Cost_Paid_fraction_of_NAV": float(turnover["cost"].sum())}
    summary.loc[STRAT, "Gross_CAGR"] = costs["Gross_CAGR"]
    summary.loc[STRAT, "Net_CAGR"] = costs["Net_CAGR"]
    summary.loc[STRAT, "Annual_Cost_Drag"] = costs["Annual_Cost_Drag"]
    summary.T.to_csv(perf / "summary.csv")
    yearly_turnover.rename("one_way_turnover").to_csv(perf / "turnover_by_year.csv")

    # --- annual / monthly returns -------------------------------------------------------
    ann = pd.DataFrame({BENCH: M.annual_returns(levels[BENCH]), STRAT: M.annual_returns(levels[STRAT]),
                        f"{STRAT}_Net": M.annual_returns(levels[f"{STRAT}_Net"])})
    ann["Excess"] = ann[STRAT] - ann[BENCH]
    ann.index = ann.index.astype(str)
    ann.to_csv(perf / "annual_returns.csv", float_format="%.6f")
    mon = pd.DataFrame({BENCH: M.monthly_returns(levels[BENCH]), STRAT: M.monthly_returns(levels[STRAT])})
    mon["Excess"] = mon[STRAT] - mon[BENCH]
    mon.index = mon.index.astype(str)
    mon.to_csv(perf / "monthly_returns.csv", float_format="%.6f")

    # --- rolling returns ------------------------------------------------------------------
    roll = {}
    for yrs, annualise, name in ((1, False, "1y_return"), (3, True, "3y_cagr"), (5, True, "5y_cagr")):
        rs_ = M.rolling_return(levels[STRAT], yrs, annualise)
        rb_ = M.rolling_return(levels[BENCH], yrs, annualise)
        df = pd.DataFrame({"Strategy": rs_, "CSI300": rb_}).dropna()
        df["Excess"] = df["Strategy"] - df["CSI300"]
        df.to_csv(perf / f"rolling_{name}.csv", float_format="%.6f")
        roll[name] = df
    rolling_stats = {f"rolling_{k}_excess_win_rate": float((v["Excess"] > 0).mean()) for k, v in roll.items()}
    rolling_stats.update({f"rolling_{k}_median_excess": float(v["Excess"].median()) for k, v in roll.items()})

    # --- drawdowns, cycles, momentum crashes ------------------------------------------------
    dd_stats = pd.DataFrame({BENCH: M.max_drawdown(levels[BENCH]), STRAT: M.max_drawdown(levels[STRAT])})
    dd_stats.to_csv(perf / "drawdown_stats.csv")
    cycles = M.cycle_table(levels[STRAT], levels[BENCH], acfg["market_cycles"], periods)
    cycles.to_csv(perf / "market_cycles.csv", index=False, float_format="%.6f")
    crashes = M.momentum_crashes(levels[STRAT], levels[BENCH])
    crashes.to_csv(perf / "momentum_crashes.csv", index=False, float_format="%.6f")
    rel_dd = M.max_drawdown(rs["Strategy_to_CSI300_Ratio"])

    # --- charts -----------------------------------------------------------------------------
    P.growth_of_100(levels[[BENCH, STRAT]], charts / "growth_of_100.png", log_scale=False, net=levels[f"{STRAT}_Net"])
    P.growth_of_100(levels[[BENCH, STRAT]], charts / "growth_of_100_log.png", log_scale=True, net=levels[f"{STRAT}_Net"])
    P.relative_strength(rs, charts / "relative_strength.png")
    P.drawdowns(levels[[BENCH, STRAT]], charts / "drawdown_comparison.png")
    P.annual_returns_bar(ann, charts / "annual_returns.png")
    P.rolling_chart({"Strategy 1Y": roll["1y_return"]["Strategy"], "CSI300 1Y": roll["1y_return"]["CSI300"]},
                    "Rolling 1-year return", "1Y return", charts / "rolling_1y_return.png")
    P.rolling_chart({"Strategy 3Y CAGR": roll["3y_cagr"]["Strategy"], "CSI300 3Y CAGR": roll["3y_cagr"]["CSI300"]},
                    "Rolling 3-year CAGR", "3Y CAGR", charts / "rolling_3y_cagr.png")
    P.rolling_chart({"Strategy 5Y CAGR": roll["5y_cagr"]["Strategy"], "CSI300 5Y CAGR": roll["5y_cagr"]["CSI300"]},
                    "Rolling 5-year CAGR", "5Y CAGR", charts / "rolling_5y_cagr.png")
    P.rolling_chart({"1Y Excess Return": roll["1y_return"]["Excess"], "3Y Excess CAGR": roll["3y_cagr"]["Excess"],
                     "5Y Excess CAGR": roll["5y_cagr"]["Excess"]},
                    "Rolling excess return: Strategy minus CSI 300", "Excess", charts / "rolling_excess_return.png")
    P.turnover_chart(turnover, charts / "turnover.png")
    # candlesticks: monthly (default for the long-run report), weekly, daily(last 2 years)
    ma_s = P.moving_averages(s_ohlc["Close"])
    ma_b = P.moving_averages(b_ohlc["Close"])
    so_m, bo_m = ohlc["M"]
    P.candlestick_chart(so_m, "CSI300 S&P Financial Viability + SPMO — monthly candles", charts / "strategy_candlestick.png", log_scale=True)
    P.candlestick_chart(bo_m, "CSI 300 — monthly candles", charts / "csi300_candlestick.png", log_scale=True)
    P.candlestick_comparison(so_m, bo_m, charts / "candlestick_comparison.png", "Monthly")
    ma_cols = ["MA50", "MA100"]
    P.candlestick_chart(so_m, "Strategy — monthly candles + MA50 / MA100 (daily-close moving averages)",
                        charts / "strategy_candlestick_ma.png", ma=ma_s[ma_cols].reindex(so_m.index), log_scale=True)
    P.candlestick_chart(bo_m, "CSI 300 — monthly candles + MA50 / MA100 (daily-close moving averages)",
                        charts / "csi300_candlestick_ma.png", ma=ma_b[ma_cols].reindex(bo_m.index), log_scale=True)
    P.candlestick_comparison(so_m, bo_m, charts / "candlestick_comparison_ma.png", "Monthly",
                             ma_s=ma_s[ma_cols].reindex(so_m.index), ma_b=ma_b[ma_cols].reindex(bo_m.index))
    so_w, bo_w = ohlc["W"]
    P.candlestick_comparison(so_w, bo_w, charts / "candlestick_comparison_weekly.png", "Weekly")
    n_daily = 500
    P.candlestick_comparison(s_ohlc.iloc[-n_daily:], b_ohlc.iloc[-n_daily:], charts / "candlestick_comparison_daily.png", "Daily",
                             ma_s=ma_s.iloc[-n_daily:], ma_b=ma_b.iloc[-n_daily:])
    P.candlestick_chart(s_ohlc.iloc[-n_daily:], "Strategy — daily candles (last 500 days) + MA20/50/100/200",
                        charts / "strategy_candlestick_daily_ma.png", ma=ma_s.iloc[-n_daily:])
    P.candlestick_chart(b_ohlc.iloc[-n_daily:], "CSI 300 — daily candles (last 500 days) + MA20/50/100/200",
                        charts / "csi300_candlestick_daily_ma.png", ma=ma_b.iloc[-n_daily:])
    ma_s.to_csv(perf / "strategy_moving_averages.csv", float_format="%.4f")
    ma_b.to_csv(perf / "csi300_moving_averages.csv", float_format="%.4f")

    results = {"levels": levels, "summary": summary, "annual": ann, "monthly": mon, "rolling": roll, "rolling_stats": rolling_stats,
               "drawdowns": dd_stats, "cycles": cycles, "crashes": crashes, "turnover": turnover, "yearly_turnover": yearly_turnover,
               "costs": costs, "relative_dd": rel_dd, "rs": rs}
    with open(perf / "key_figures.json", "w", encoding="utf-8") as fh:
        json.dump({"summary": json.loads(summary.to_json(date_format="iso")), "costs": costs, "rolling_stats": rolling_stats,
                   "relative_strength_drawdown": {k: (str(v) if not isinstance(v, (int, float, type(None))) else v) for k, v in rel_dd.items()}},
                  fh, ensure_ascii=False, indent=1, default=str)
    return results
