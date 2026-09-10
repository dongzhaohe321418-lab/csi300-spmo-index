"""End-to-end pipeline stages used by main.py and the scripts/ entry points."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd

from . import analysis, audit, backtest, prices, report
from .data import DataHub
from .universe import build_universe, load_events
from .utils import TradingCalendar, abs_path, ensure_dirs, load_config, load_env, setup_logging

log = logging.getLogger("pipeline")


class Context:
    def __init__(self, config_path: str = "config.yaml", refresh: bool = False):
        self.cfg = load_config(config_path)
        load_env(self.cfg)
        ensure_dirs(self.cfg)
        self.refresh = refresh
        self.hub = DataHub(self.cfg)
        idx = self.hub.index_daily(self.cfg["data"]["parent_index_code"], refresh=refresh)
        self.hub.index_daily(self.cfg["data"]["parent_index_tr_code"], refresh=refresh)
        self.cal = TradingCalendar(idx["date"])
        self.index_daily = idx
        self.processed = self.hub.processed
        self.prices_dir = self.processed / "prices"
        self.perf_dir = abs_path(self.cfg, self.cfg["output"]["performance_dir"])

    # ------------------------------------------------------------------ stages
    def stage_universe(self) -> pd.DataFrame:
        t = time.time()
        intervals, aud, events = build_universe(self.hub, self.cal, refresh=self.refresh)
        log.info("universe: %d securities, %d intervals, %d events, validated in %.0fs", intervals["code"].nunique(), len(intervals),
                 len(events), time.time() - t)
        return intervals

    def load_intervals(self) -> pd.DataFrame:
        p = self.processed / "csi300_membership_intervals.csv"
        if not p.exists():
            return self.stage_universe()
        df = pd.read_csv(p, dtype={"code": str}, parse_dates=["start_date", "end_date"])
        return df

    def stage_prices(self, intervals: pd.DataFrame, rebuild: bool = False) -> pd.DataFrame:
        codes = sorted(set(intervals["code"]))
        status_path = self.processed / "price_adjustment_status.csv"
        if status_path.exists() and not rebuild and len(list(self.prices_dir.glob("*.parquet"))) >= len(codes) - 5:
            return pd.read_csv(status_path, dtype={"code": str})
        t = time.time()
        status = prices.build_all(self.hub, codes, self.cal.days, self.prices_dir)
        status.to_csv(status_path, index=False)
        log.info("prices: %d ok, %d without data (%.0fs)", (status["status"] == "ok").sum(), (status["status"] != "ok").sum(), time.time() - t)
        return status

    def load_panel(self) -> dict:
        t = time.time()
        panel = prices.load_panel(self.prices_dir, self.cal.days)
        log.info("panel: %d dates x %d securities (%.0fs)", *panel["adj_close_pr"].shape, time.time() - t)
        return panel

    def stage_backtest(self, intervals: pd.DataFrame, panel: dict) -> dict:
        codes = sorted(set(intervals["code"]))
        t = time.time()
        qtabs = backtest.load_quarterlies(self.hub, codes, self.cfg["financial_viability"]["field_priority"])
        log.info("financials: quarterly tables for %d/%d securities (%.0fs)", len(qtabs), len(codes), time.time() - t)
        names = backtest.name_map(self.hub, intervals, codes)
        backtest.validate_fmc_proxy(self.cfg, self.hub, panel, intervals)
        bt = backtest.run(self.cfg, self.hub, self.cal, intervals, panel, qtabs, names)
        schedule = backtest.rebalance_schedule(self.cal, self.cfg, panel["adj_close_pr"].index[-1])
        aud = audit.run(self.cfg, intervals, schedule, self.cfg["momentum"]["min_traded_days"])
        aud.to_csv(abs_path(self.cfg, self.cfg["output"]["performance_dir"]) / "eligibility_audit.csv", index=False)
        log.info("eligibility audit passed for %d reference dates", len(aud))
        # persist the raw engine outputs for scripts that only redo analysis/report
        out = abs_path(self.cfg, self.cfg["output"]["performance_dir"])
        bt["strategy_pr"]["levels"].to_csv(out / "_engine_strategy_pr.csv")
        bt["strategy_tr"]["levels"].to_csv(out / "_engine_strategy_tr.csv")
        bt["benchmark"].to_csv(out / "_engine_benchmark.csv")
        bt["benchmark_tr"].to_csv(out / "_engine_benchmark_tr.csv")
        bt["strategy_pr"]["turnover"].to_csv(out / "_engine_turnover.csv", index=False)
        return bt

    def load_backtest(self) -> dict:
        out = abs_path(self.cfg, self.cfg["output"]["performance_dir"])
        rd = lambda n: pd.read_csv(out / n, index_col=0, parse_dates=True)  # noqa: E731
        to = pd.read_csv(out / "_engine_turnover.csv", parse_dates=["reference_date", "effective_date"])
        return {"strategy_pr": {"levels": rd("_engine_strategy_pr.csv"), "turnover": to},
                "strategy_tr": {"levels": rd("_engine_strategy_tr.csv"), "turnover": to},
                "benchmark": rd("_engine_benchmark.csv"), "benchmark_tr": rd("_engine_benchmark_tr.csv"),
                "summary": pd.read_csv(out / "rebalance_summary.csv")}

    def stage_analysis(self, bt: dict) -> dict:
        t = time.time()
        an = analysis.run(self.cfg, bt)
        log.info("analysis & charts done (%.0fs)", time.time() - t)
        return an

    def stage_report(self, bt: dict, an: dict, intervals: pd.DataFrame) -> Path:
        events = load_events(self.processed / "csi300_adjustment_events.json")
        ev = [e for e in events if e.kind != "full_list" and not e.pending]
        universe_info = {"n_events": len(ev), "n_regular": sum(e.kind == "regular" for e in ev),
                         "n_temporary": sum(e.kind != "regular" for e in ev), "n_codes": intervals["code"].nunique()}
        data_info = self.data_info(intervals)
        path = report.build(self.cfg, bt, an, universe_info, data_info)
        self.write_readme_block(an, bt)
        log.info("report written to %s", path)
        return path

    # ------------------------------------------------------------------ helpers
    def data_info(self, intervals: pd.DataFrame) -> dict:
        idx = self.index_daily
        sources = [
            {"Item": "Historical CSI 300 constituents", "Source": "CSIndex official announcements (regular reviews, temporary adjustments, 2005 full list)",
             "Coverage": f"{intervals['start_date'].min().date()} → {idx['date'].max().date()}"},
            {"Item": "Historical CSI 300 weights", "Source": "Approximated by float-adjusted market cap proxy (close × listed A-shares); cross-checked against the official 2026-08-31 weight file", "Coverage": "all reference dates"},
            {"Item": "Stock daily OHLC / volume / amount", "Source": "Sina Finance (incl. delisted securities)", "Coverage": "listing → delisting / latest"},
            {"Item": "Adjusted prices & daily returns", "Source": "Built from raw prices + exchange ex-date reference prices + Eastmoney dividend / bonus / rights records", "Coverage": "same"},
            {"Item": "Float-adjusted market cap", "Source": "Eastmoney share-structure history (已上市流通A股) × close", "Coverage": "same"},
            {"Item": "Quarterly financials & announcement dates", "Source": "Eastmoney F10 income statements (NOTICE_DATE, CONTINUED_NETPROFIT, NETPROFIT)", "Coverage": "listed companies; delisted ones only where still served"},
            {"Item": "Corporate actions", "Source": "Eastmoney 分红送配 detail + market-wide 配股 table", "Coverage": "listed companies"},
            {"Item": "CSI 300 index OHLC (price & total return)", "Source": "CSIndex official (000300, H00300)", "Coverage": f"{idx['date'].min().date()} → {idx['date'].max().date()}"},
        ]
        notes = ("All raw downloads are cached under `data/raw/` and the derived tables under `data/processed/`. "
                 "API keys are not required for the default free sources; an optional Tushare token can be supplied via `.env` "
                 "(`TUSHARE_TOKEN`) and is never committed.\n")
        wv = self.perf_dir / "weight_proxy_validation_summary.csv"
        wtxt = ""
        if wv.exists():
            st = pd.read_csv(wv, header=None, index_col=0)[1]
            wtxt = (f" Against the official CSI 300 weight file of {st['as_of']}, the proxy weights have correlation "
                    f"{float(st['correlation']):.2f}, mean absolute deviation {100 * float(st['mean_abs_diff']):.2f} pp and an aggregate "
                    f"active share of {100 * float(st['sum_abs_diff (active share vs official)']):.0f}% (`weight_proxy_validation.csv`); the "
                    f"proxy over-weights state-controlled large caps whose listed A-shares are largely non-free-float.")
        caveats = (
            "- **Float-adjusted market cap** is approximated by listed A-shares × close; CSIndex's security-level free-float factors are "
            "not public for history, so parent weights and SPMO weights are approximations." + wtxt + " An optional refinement that "
            "subtracts non-free-float holders (≥5%, non-institutional) from Sina's quarterly top-10 shareholder history with CSIndex's "
            "tiering is implemented (`scripts/download_data.py --sources holders`) and is applied automatically once shareholder data cover "
            "≥90% of the securities.\n"
            "- **Announcement dates** before 2010 are not reliable in the free income-statement source (lag ≈ 13 months); they are replaced "
            "by the statutory filing deadline, which delays the availability of some reports by a few weeks relative to their true "
            "publication (conservative).\n"
            "- **Financial statements of delisted companies** are not served by Eastmoney's F10 for most delisted securities, so such "
            "securities fail Financial Viability (no data) while they were members — a conservative treatment that slightly under-states the "
            "eligible universe in the early years.\n"
            "- **Restatements**: the income-statement source stores the latest available figure per period together with the *first* "
            "announcement date; restated values may differ from the originally published numbers.\n"
            "- **Suspended securities** are carried at their last price; at a rebalance they are bought/sold at that price (in practice the trade "
            "would occur on resumption).\n"
            "- **Strategy OHLC** aggregates constituent highs/lows with fixed index shares, which slightly overstates the intraday range; Open and Close are exact.\n"
            "- The 2008-07 regular review's main announcement is mis-filed in the CSIndex archive; the change list was taken from the same-day "
            "sector-index companion document (identical construction validated on 2008-12).\n"
        )
        return {"sources": sources, "notes": notes, "caveats": caveats}

    def write_readme_block(self, an: dict, bt: dict) -> None:
        """Refresh the auto-generated performance block in README.md."""
        readme = Path(self.cfg["_root"]) / "README.md"
        if not readme.exists():
            return
        summary = an["summary"]
        tab = report.core_table(summary)
        lev = an["levels"]
        block = ["<!-- AUTO-RESULTS-START -->",
                 f"*Backtest {lev.index[0].date()} → {lev.index[-1].date()} (auto-generated by `python main.py`)*", "",
                 tab.to_markdown(), "",
                 "![Growth of ¥100](output/charts/growth_of_100.png)", "",
                 "![Strategy vs CSI300 candlesticks](output/charts/candlestick_comparison.png)", "",
                 "<!-- AUTO-RESULTS-END -->"]
        txt = readme.read_text(encoding="utf-8")
        s, e = txt.find("<!-- AUTO-RESULTS-START -->"), txt.find("<!-- AUTO-RESULTS-END -->")
        if s >= 0 and e >= 0:
            txt = txt[:s] + "\n".join(block) + txt[e + len("<!-- AUTO-RESULTS-END -->"):]
            readme.write_text(txt, encoding="utf-8")


def run_all(config_path: str = "config.yaml", refresh: bool = False, skip_download: bool = False, log_level: str = "INFO") -> None:
    setup_logging(log_level)
    t0 = time.time()
    ctx = Context(config_path, refresh=refresh)
    if not skip_download:
        from scripts.download_data import download_all  # lazy import: heavy network step

        download_all(ctx.hub, ctx.cal, refresh=refresh)
    intervals = ctx.stage_universe() if (refresh or not (ctx.processed / "csi300_membership_intervals.csv").exists()) else ctx.load_intervals()
    ctx.stage_prices(intervals, rebuild=refresh)
    panel = ctx.load_panel()
    bt = ctx.stage_backtest(intervals, panel)
    an = ctx.stage_analysis(bt)
    ctx.stage_report(bt, an, intervals)
    log.info("pipeline finished in %.0fs", time.time() - t0)
