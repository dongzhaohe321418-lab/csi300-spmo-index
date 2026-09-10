"""Rebalance loop: universe -> financial viability -> momentum -> selection -> weights -> index.

Every reference date produces output/scores/YYYY-MM-DD.csv (all CSI 300 members on that
date with every intermediate quantity) and, once implemented, output/holdings/YYYY-MM-DD.csv
(the constituents and weights in force from the close of the implementation date).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .data import DataHub, NoData
from .financials import quarterly_table, viability_table
from .index_engine import IndexEngine, Rebalance, benchmark_levels
from .momentum import compute_momentum
from .selection import select_constituents
from .universe import members_at
from .utils import TradingCalendar, abs_path
from .weighting import compute_weights

log = logging.getLogger(__name__)

SCORE_COLUMNS = ["reference_date", "ticker", "name", "csi300_member", "quarter_net_income", "four_quarter_net_income",
                 "financial_eligible", "momentum_return", "volatility", "RAM", "z_score", "momentum_score", "rank",
                 "selected", "csi300_weight", "raw_weight", "final_weight"]


def rebalance_schedule(cal: TradingCalendar, cfg: dict, data_end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """(reference_date, implementation_date) pairs: last trading day of Feb/Aug -> 3rd Friday of Mar/Sep."""
    first_ref = pd.Timestamp(cfg["rebalance"]["first_reference_date"])
    out = []
    for y in range(first_ref.year, data_end.year + 1):
        for m_ref, m_eff in zip(cfg["rebalance"]["reference_months"], cfg["rebalance"]["effective_months"]):
            if pd.Timestamp(year=y, month=m_ref, day=1) > data_end:
                continue
            ref = cal.month_end(y, m_ref)
            if ref < first_ref or ref > data_end:
                continue
            eff = cal.third_friday(y, m_eff)
            out.append((ref, eff))
    return out


def load_quarterlies(hub: DataHub, codes, field_priority) -> dict[str, pd.DataFrame]:
    out = {}
    for c in codes:
        try:
            inc = hub.income(c)
        except NoData:
            continue
        out[c] = quarterly_table(inc, field_priority=tuple(field_priority))
    return out


def name_map(hub: DataHub, intervals: pd.DataFrame, codes) -> dict[str, str]:
    names = {}
    for c, n in zip(intervals["code"], intervals["name"]):
        if isinstance(n, str) and n:
            names.setdefault(c, n)
    cur = hub.csi.current_constituents()
    names.update(dict(zip(cur["code"], cur["name"])))
    return names


def validate_fmc_proxy(cfg: dict, hub: DataHub, panel: dict[str, pd.DataFrame], intervals: pd.DataFrame) -> pd.DataFrame:
    """Compare the float-cap proxy weights with the latest official CSI 300 weight file."""
    w = hub.csi.current_weights()
    asof = pd.Timestamp(w["as_of"].iloc[0])
    if asof not in panel["fmc"].index:
        asof = panel["fmc"].index[panel["fmc"].index.searchsorted(asof, side="right") - 1]
    members = sorted(members_at(intervals, asof))
    fmc = panel["fmc"].loc[asof].reindex(members)
    proxy = (fmc / fmc.sum()).rename("proxy_weight")
    cmp = w.set_index("code")["weight"].rename("official_weight").to_frame().join(proxy, how="outer")
    cmp["diff"] = cmp["proxy_weight"] - cmp["official_weight"]
    stats = {
        "as_of": asof.date(), "n": int(cmp["official_weight"].notna().sum()),
        "correlation": float(cmp[["official_weight", "proxy_weight"]].corr().iloc[0, 1]),
        "mean_abs_diff": float(cmp["diff"].abs().mean()), "max_abs_diff": float(cmp["diff"].abs().max()),
        "sum_abs_diff (active share vs official)": float(cmp["diff"].abs().sum() / 2),
    }
    out = abs_path(cfg, cfg["output"]["performance_dir"])
    cmp.sort_values("official_weight", ascending=False).to_csv(out / "weight_proxy_validation.csv", float_format="%.6f")
    pd.Series(stats).to_csv(out / "weight_proxy_validation_summary.csv", header=False)
    log.info("FMC proxy vs official weights (%s): corr=%.3f, mean|diff|=%.3f%%, max|diff|=%.2f%%", stats["as_of"],
             stats["correlation"], 100 * stats["mean_abs_diff"], 100 * stats["max_abs_diff"])
    return cmp


def run(cfg: dict, hub: DataHub, cal: TradingCalendar, intervals: pd.DataFrame, panel: dict[str, pd.DataFrame],
        qtabs: dict[str, pd.DataFrame], names: dict[str, str]) -> dict:
    scores_dir = abs_path(cfg, cfg["output"]["scores_dir"])
    hold_dir = abs_path(cfg, cfg["output"]["holdings_dir"])
    perf_dir = abs_path(cfg, cfg["output"]["performance_dir"])
    for d in (scores_dir, hold_dir, perf_dir):
        d.mkdir(parents=True, exist_ok=True)
    for d in (scores_dir, hold_dir):          # remove per-date files of earlier runs (stale dates)
        for f in d.glob("*.csv"):
            f.unlink()
    data_end = panel["adj_close_pr"].index[-1]
    schedule = rebalance_schedule(cal, cfg, data_end)
    fv = cfg["financial_viability"]
    mcfg, scfg, wcfg = cfg["momentum"], cfg["selection"], cfg["weighting"]

    rebalances: list[Rebalance] = []
    current: set = set()
    summaries = []
    for ref, eff in schedule:
        members = sorted(members_at(intervals, ref))
        if len(members) != cfg["universe"]["expected_size"]:
            raise RuntimeError(f"{ref.date()}: {len(members)} CSI300 members (expected {cfg['universe']['expected_size']})")
        fin = viability_table(qtabs, members, ref, require_latest_positive=fv["require_latest_quarter_positive"],
                              require_4q_positive=fv["require_trailing_4q_sum_positive"])
        mom = compute_momentum(panel["adj_close_pr"], panel["ret_pr"], panel["traded"], ref, cal, codes=members,
                               lookback_months=mcfg["lookback_months"], skip_months=mcfg["skip_months"],
                               min_traded_days=mcfg["min_traded_days"])
        df = pd.DataFrame({"code": members})
        df = df.merge(fin, on="code", how="left").merge(mom, on="code", how="left")
        df["momentum_eligible"] = df["momentum_eligible"].fillna(False).astype(bool)
        df["eligible"] = df["financial_eligible"].fillna(False).astype(bool) & df["momentum_eligible"]
        # market caps on the reference date
        fmc = panel["fmc"].loc[ref].reindex(members)
        df["fmc"] = fmc.values
        fmc_ok = fmc.dropna()
        df["csi300_weight"] = (fmc / fmc_ok.sum()).values
        missing_fmc = df["eligible"] & df["fmc"].isna()
        if missing_fmc.any():
            log.warning("%s: %d eligible securities without market cap -> excluded", ref.date(), int(missing_fmc.sum()))
            df.loc[missing_fmc, "eligible"] = False
        sel = select_constituents(df, current, top_fraction=scfg["top_fraction"], auto_frac=scfg["buffer"]["auto_select_fraction"],
                                  retain_frac=scfg["buffer"]["retain_fraction"], rounding=scfg["rounding"])
        chosen = sel[sel["selected"]].copy()
        wdf = compute_weights(chosen, cap_absolute=wcfg["cap_absolute"], cap_multiple=wcfg["cap_multiple_of_parent_weight"],
                              max_iter=wcfg["max_cap_iterations"])
        # parent membership has priority: a selected security that leaves the CSI 300 between
        # the reference date and the implementation date is not implemented (weight 0)
        mem_eff = members_at(intervals, eff) if eff <= data_end else set(members)
        left = [c for c in wdf["code"] if c not in mem_eff]
        if left:
            log.warning("%s: %s left the CSI 300 before implementation on %s -> weight set to 0, others renormalised",
                        ref.date(), left, eff.date())
            wdf.loc[wdf["code"].isin(left), "final_weight"] = 0.0
            wdf["final_weight"] = wdf["final_weight"] / wdf["final_weight"].sum()
            wdf = wdf[wdf["final_weight"] > 0].copy()
            sel.loc[sel["code"].isin(left), "selected"] = False
        sel["raw_weight"] = np.nan
        sel["final_weight"] = 0.0
        sel.loc[wdf.index, "raw_weight"] = wdf["raw_weight"]
        sel.loc[wdf.index, "final_weight"] = wdf["final_weight"]
        sel["weight_cap"] = np.nan
        sel.loc[wdf.index, "weight_cap"] = wdf["weight_cap"]
        # --- scores file --------------------------------------------------------
        sel["reference_date"] = ref.strftime("%Y-%m-%d")
        sel["ticker"] = sel["code"]
        sel["name"] = sel["code"].map(names).fillna("")
        sel["csi300_member"] = True
        sel["selected"] = sel["selected"].astype(bool)
        extra = ["fin_latest_period", "fin_latest_available", "fin_field", "fin_reason", "mom_start_date", "mom_end_date",
                 "traded_days", "momentum_eligible", "mom_reason", "eligible", "fmc", "weight_cap", "buffer_retained"]
        out = sel[SCORE_COLUMNS + extra].sort_values(["selected", "rank"], ascending=[False, True])
        out.to_csv(scores_dir / f"{ref:%Y-%m-%d}.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
        implemented = eff <= data_end
        meta = {"reference_date": str(ref.date()), "implementation_date": str(eff.date()), "implemented": bool(implemented),
                "n_members": len(members), "n_financial_eligible": int(df["financial_eligible"].fillna(False).sum()),
                "n_momentum_eligible": int(df["momentum_eligible"].sum()), "n_eligible": int(df["eligible"].sum()),
                "n_target": sel.attrs["n_target"], "n_selected": int(sel["selected"].sum()),
                "n_buffer_retained": int(sel["buffer_retained"].sum()), **{k: v for k, v in wdf.attrs.items()}}
        summaries.append(meta)
        # --- holdings file --------------------------------------------------------
        hold = wdf.copy()
        hold["Effective_Date"] = eff.strftime("%Y-%m-%d")
        hold["Reference_Date"] = ref.strftime("%Y-%m-%d")
        hold = hold.assign(Ticker=hold["code"], Name=hold["code"].map(names).fillna(""))
        hold_out = hold[["Reference_Date", "Effective_Date", "Ticker", "Name", "momentum_score", "rank", "csi300_weight", "raw_weight",
                         "final_weight", "fmc", "weight_cap", "capped", "buffer_retained"]].rename(
            columns={"momentum_score": "Momentum_Score", "rank": "Rank", "csi300_weight": "CSI300_Weight", "raw_weight": "Raw_Weight",
                     "final_weight": "Final_Weight", "fmc": "Float_Market_Cap", "weight_cap": "Weight_Cap", "capped": "Capped",
                     "buffer_retained": "Buffer_Retained"}).sort_values("Final_Weight", ascending=False)
        fname = f"{eff:%Y-%m-%d}.csv" if implemented else f"{eff:%Y-%m-%d}_pending.csv"
        hold_out.to_csv(hold_dir / fname, index=False, encoding="utf-8-sig", float_format="%.10g")
        log.info("%s -> %s: members=300 fin_ok=%d mom_ok=%d eligible=%d target=%d selected=%d capped=%d mult=%.0f%s",
                 ref.date(), eff.date(), meta["n_financial_eligible"], meta["n_momentum_eligible"], meta["n_eligible"],
                 meta["n_target"], meta["n_selected"], meta["n_capped"], meta["cap_multiple_used"],
                 "" if implemented else " (pending implementation)")
        if implemented:
            rebalances.append(Rebalance(ref, eff, wdf.set_index("code")["final_weight"], meta))
            current = set(wdf["code"])
    pd.DataFrame(summaries).to_csv(perf_dir / "rebalance_summary.csv", index=False)

    # --- index calculation ----------------------------------------------------------
    engine = IndexEngine(panel, cal, intervals, cfg)
    base = float(cfg["project"]["base_level"])
    res_pr = engine.run(rebalances, basis="pr", base_level=base)
    res_tr = engine.run(rebalances, basis="tr", base_level=base)
    start = rebalances[0].effective_date
    bench = benchmark_levels(hub.index_daily(cfg["data"]["parent_index_code"]), start, base)
    bench_tr = benchmark_levels(hub.index_daily(cfg["data"]["parent_index_tr_code"]), start, base)
    return {"rebalances": rebalances, "strategy_pr": res_pr, "strategy_tr": res_tr, "benchmark": bench,
            "benchmark_tr": bench_tr, "summary": pd.DataFrame(summaries)}
