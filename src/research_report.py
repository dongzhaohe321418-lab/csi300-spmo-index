"""完整研究报告（中文）生成器：output/report/research_report_zh.md (+ .docx)。

所有数字均由程序从 output/ 与 data/processed/ 读取或由 src.research 现场计算，报告文字中不手写任何回测数字；
唯一的外部数字是 GitHub 项目 HSI300-Momentum-Strategy 在其 README 中公布的结果（转录自该仓库 result.PNG）。
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from . import prices as PR
from . import research as R
from . import research_charts as RC
from .data import DataHub
from .universe import load_events
from .utils import TradingCalendar, abs_path

log = logging.getLogger(__name__)

# 该仓库 README 公布的结果（result.PNG，2019-01-01 → 2025-12-20，BaoStock 数据）
HSMO_REPORTED = {
    "annual": {2019: (0.4095, 0.3795), 2020: (0.8591, 0.2721), 2021: (0.1111, -0.0520), 2022: (-0.0867, -0.2163),
               2023: (-0.0654, -0.1138), 2024: (0.3082, 0.1468), 2025: (0.1782, 0.1609)},
    "summary": {"累计收益": (2.8362, 0.5383), "年化收益": (0.2218, 0.0663), "年化波动": (0.2442, 0.1912),
                "夏普比率": (0.83, 0.24), "最大回撤": (-0.3351, -0.4560)},
    "url": "https://github.com/dongzhaohe321418-lab/HSI300-Momentum-Strategy",
}
HSMO_WINDOW = ("2019-01-01", "2025-12-20")


# --------------------------------------------------------------------------- helpers
def pct(x, d: int = 2, sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%"


def num(x, d: int = 2) -> str:
    return f"{x:,.{d}f}"


def md_table(df: pd.DataFrame, index: bool = False, floatfmt: str = ".2f") -> str:
    return df.to_markdown(index=index, floatfmt=floatfmt)


def _fmt_pct_cols(df: pd.DataFrame, cols, d=2, sign=False) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].map(lambda v: pct(v, d, sign) if pd.notna(v) else "")
    return out


# --------------------------------------------------------------------------- collection
def collect(cfg: dict, hub: DataHub, cal: TradingCalendar) -> dict:
    t0 = time.time()
    perf = abs_path(cfg, cfg["output"]["performance_dir"])
    processed = hub.processed
    research_dir = abs_path(cfg, "output/research")
    research_dir.mkdir(parents=True, exist_ok=True)
    charts = abs_path(cfg, cfg["output"]["charts_dir"]) / "research"

    ctx: dict = {"cfg": cfg, "generated": pd.Timestamp.today().strftime("%Y-%m-%d")}
    levels = R.load_levels(cfg)
    ctx["levels"] = levels
    ctx["summary"] = pd.read_csv(perf / "summary.csv", index_col=0)
    ctx["key"] = json.load(open(perf / "key_figures.json"))
    ctx["annual"] = pd.read_csv(perf / "annual_returns.csv")
    ctx["cycles"] = pd.read_csv(perf / "market_cycles.csv")
    ctx["crashes"] = pd.read_csv(perf / "momentum_crashes.csv")
    ctx["turnover_by_year"] = pd.read_csv(perf / "turnover_by_year.csv")
    ctx["rebalance"] = pd.read_csv(perf / "rebalance_summary.csv", parse_dates=["reference_date", "implementation_date"])
    ctx["audit"] = pd.read_csv(perf / "eligibility_audit.csv")
    ctx["wval"] = pd.read_csv(perf / "weight_proxy_validation_summary.csv", header=None, index_col=0)[1]
    ctx["dd"] = pd.read_csv(perf / "drawdown_stats.csv", index_col=0)
    ctx["rs_dd"] = ctx["key"]["relative_strength_drawdown"]

    # universe & data info
    intervals = pd.read_csv(processed / "csi300_membership_intervals.csv", dtype={"code": str}, parse_dates=["start_date", "end_date"])
    events = [e for e in load_events(processed / "csi300_adjustment_events.json") if e.kind != "full_list" and not e.pending]
    ctx["universe"] = {"n_codes": intervals["code"].nunique(), "n_intervals": len(intervals), "n_events": len(events),
                       "n_regular": sum(e.kind == "regular" for e in events), "n_temporary": sum(e.kind != "regular" for e in events),
                       "first_event": min(e.effective_date for e in events), "last_event": max(e.effective_date for e in events)}
    status = pd.read_csv(processed / "price_adjustment_status.csv", dtype={"code": str})
    ctx["price_status"] = status["status"].value_counts().to_dict()
    fmc_method = (processed / "fmc_method.txt").read_text(encoding="utf-8").strip() if (processed / "fmc_method.txt").exists() else ""
    ctx["fmc_method"] = fmc_method
    idx = hub.index_daily(cfg["data"]["parent_index_code"])
    ctx["index_span"] = (idx["date"].min().date(), idx["date"].max().date(), len(idx))

    # analytics
    strat, bench = levels["CSI300_SP_FV_SPMO"], levels["CSI300"]
    ctx["monthly"] = R.monthly_distribution(strat, bench)
    ctx["rolling"] = R.rolling_beta_te(strat, bench)
    ctx["tails"] = R.tail_stats(strat, bench)
    ctx["holdings_chars"] = R.holdings_characteristics(cfg)
    ctx["persistence"] = R.persistence(cfg)
    ctx["funnel"] = R.fv_funnel(cfg)
    sector_map = pd.read_csv(processed / "csi300_sector_map_current.csv", dtype={"code": str})
    ctx["sector_map"] = sector_map
    ctx["official_weights"] = hub.csi.current_weights()
    ctx["sector_exposure"] = R.sector_exposure(cfg, sector_map, ctx["official_weights"])
    hold_dir = abs_path(cfg, cfg["output"]["holdings_dir"])
    hfiles = sorted(hold_dir.glob("*.csv"))
    ctx["latest_holdings"] = pd.read_csv([p for p in hfiles if "pending" not in p.name][-1], dtype={"Ticker": str})
    pend = [p for p in hfiles if "pending" in p.name]
    ctx["pending_holdings"] = pd.read_csv(pend[-1], dtype={"Ticker": str}) if pend else None

    # engine-based attribution and the HSMO replication (need the price panel)
    panel = PR.load_panel(processed / "prices", cal.days)
    panel_ff = dict(panel)
    for k in ("adj_close_pr", "adj_open_pr", "adj_high_pr", "adj_low_pr", "adj_close_tr", "close"):
        panel_ff[k] = panel[k].ffill()          # replica holds suspended-until-delisting names at their last price
    attr = R.attribution_runs(cfg, cal, intervals, panel_ff)
    ctx["attr"] = attr
    off = idx.set_index("date")["close"].loc[attr.index[0]:attr.index[-1]]
    off = off / off.iloc[0] * 100
    rep = attr["CSI300_replica_proxy"]
    ctx["replica"] = {"cagr_official": M.cagr(off), "cagr_replica": M.cagr(rep), "te": M.tracking_error(rep, off),
                      "corr": M.daily_returns(rep).corr(M.daily_returns(off)), "final_official": off.iloc[-1], "final_replica": rep.iloc[-1]}
    smap = dict(zip(sector_map["code"], sector_map["sector"]))
    w0, w1 = HSMO_WINDOW
    hs = {
        "win_nocap": R.hsmo_replication(panel, intervals, w0, w1, top_n=20, cost_rate=0.0015),
        "win_cap": R.hsmo_replication(panel, intervals, w0, w1, top_n=20, cost_rate=0.0015, max_per_ind=5, sector_map=smap),
        "win_nocap_projcost": R.hsmo_replication(panel, intervals, w0, w1, top_n=20, cost_cfg=cfg["costs"]),
        "full_nocap": R.hsmo_replication(panel, intervals, levels.index[0], None, top_n=20, cost_rate=0.0015),
        "full_cap": R.hsmo_replication(panel, intervals, levels.index[0], None, top_n=20, cost_rate=0.0015, max_per_ind=5, sector_map=smap),
        "full_nocap_projcost": R.hsmo_replication(panel, intervals, levels.index[0], None, top_n=20, cost_cfg=cfg["costs"]),
    }
    ctx["hsmo"] = hs
    ctx["hsmo_levels"] = {k: v["levels"] for k, v in hs.items()}

    # figures
    ctx["fig"] = {
        "attr": RC.attribution_chart(levels, attr, charts / "rr_attribution.png"),
        "cycle": RC.cycle_cagr_chart(ctx["cycles"], charts / "rr_cycle_cagr.png"),
        "rolling": RC.rolling_chart(ctx["rolling"], charts / "rr_rolling_beta_te.png"),
        "monthly": RC.monthly_excess_chart(ctx["monthly"]["monthly_excess"], charts / "rr_monthly_excess.png"),
        "conc": RC.concentration_chart(ctx["holdings_chars"], charts / "rr_concentration.png"),
        "funnel": RC.funnel_chart(ctx["funnel"], charts / "rr_fv_funnel.png"),
        "pers": RC.persistence_chart(ctx["persistence"], charts / "rr_persistence.png"),
        "sector": RC.sector_chart(ctx["sector_exposure"], charts / "rr_sector_exposure.png"),
        "hsmo": RC.hsmo_chart(levels, ctx["hsmo_levels"], charts / "rr_hsmo_comparison.png"),
        "dd": RC.drawdown_compare_chart(levels, hs["full_nocap"]["levels"], charts / "rr_drawdown_comparison.png"),
    }

    main_charts = abs_path(cfg, cfg["output"]["charts_dir"])
    for k, f in (("fig_growth", "growth_of_100.png"), ("fig_growth_log", "growth_of_100_log.png"), ("fig_candles", "candlestick_comparison.png"),
                 ("fig_annual", "annual_returns.png"), ("fig_dd", "drawdown_comparison.png"), ("fig_rs", "relative_strength.png"), ("fig_turnover", "turnover.png")):
        ctx[k] = main_charts / f

    # persist the analytic tables
    attr.to_csv(research_dir / "attribution_levels.csv", float_format="%.6f")
    pd.DataFrame(ctx["hsmo_levels"]).to_csv(research_dir / "hsmo_replication_levels.csv", float_format="%.6f")
    ctx["holdings_chars"].to_csv(research_dir / "holdings_characteristics.csv", index=False, float_format="%.6f")
    ctx["funnel"].to_csv(research_dir / "financial_viability_funnel.csv", index=False)
    ctx["persistence"]["top"].to_csv(research_dir / "constituent_persistence.csv", index=False, encoding="utf-8-sig")
    ctx["sector_exposure"].to_csv(research_dir / "sector_exposure_current.csv", float_format="%.6f", encoding="utf-8-sig")
    ctx["rolling"].to_csv(research_dir / "rolling_beta_te.csv", float_format="%.6f")
    ctx["tails"].to_csv(research_dir / "tail_statistics.csv", index=False, float_format="%.6f")
    log.info("research analytics collected in %.0fs", time.time() - t0)
    return ctx


def build(cfg: dict, hub: DataHub, cal: TradingCalendar, latex: bool = True) -> dict[str, Path | None]:
    """Markdown + DOCX always; LaTeX source always when `latex`; PDF when a TeX engine is available
    (tectonic or latexmk; set TECTONIC_BUNDLE to pass `-b <bundle>` to tectonic)."""
    import os

    from .docx_export import markdown_to_docx
    from .research_latex import compile_pdf, write_latex
    from .research_text import write_markdown

    ctx = collect(cfg, hub, cal)
    rep_dir = abs_path(cfg, cfg["output"]["report_dir"])
    out: dict[str, Path | None] = {}
    out["md"] = write_markdown(ctx, rep_dir / "research_report_zh.md")
    out["docx"] = markdown_to_docx(out["md"], rep_dir / "research_report_zh.docx", title="沪深300 S&P Financial Viability + SPMO 动量指数研究报告")
    if latex:
        out["tex"] = write_latex(ctx, rep_dir / "research_report_zh.tex")
        extra = ["-b", os.environ["TECTONIC_BUNDLE"]] if os.environ.get("TECTONIC_BUNDLE") else None
        out["pdf"] = compile_pdf(out["tex"], extra_args=extra)
    log.info("research report written: %s", {k: str(v) for k, v in out.items()})
    return out
