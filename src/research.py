"""Extended analytics for the research report (研究报告).

Everything here is *diagnostic*: it decomposes and stress-tests the fixed strategy but never
changes it. Three groups of tools:

1. Performance analytics beyond `metrics.py`: window statistics, monthly distribution, up/down
   capture, rolling beta / tracking error, tail statistics.
2. Portfolio analytics from the per-rebalance score/holding files: concentration, size tilt,
   persistence of constituents, the Financial-Viability funnel, sector exposure (current CSI
   sector classification, latest holdings only).
3. Attribution runs with the same index engine: the selected baskets re-weighted by float cap
   and equally, and a float-cap "replica" of the CSI 300 built with the project's own weight
   proxy (measures how well the proxy tracks the official index).  Plus a faithful
   re-implementation of the rules of the GitHub project `HSI300-Momentum-Strategy` (top-20
   equal-weight, quarterly, 15 bp costs) on this project's point-in-time data, so the two
   approaches can be compared on identical data and windows.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from . import metrics as M
from .index_engine import IndexEngine, Rebalance, cost_rates
from .universe import members_at
from .utils import TradingCalendar, abs_path

log = logging.getLogger(__name__)
TD = 252


# --------------------------------------------------------------------------- #
# 1. performance analytics
# --------------------------------------------------------------------------- #
def load_levels(cfg: dict) -> pd.DataFrame:
    p = abs_path(cfg, cfg["output"]["performance_dir"]) / "index_levels.csv"
    df = pd.read_csv(p, parse_dates=["Date"]).set_index("Date")
    return df


def window(levels: pd.DataFrame, start, end=None) -> pd.DataFrame:
    s = pd.Timestamp(start)
    e = pd.Timestamp(end) if end is not None else levels.index[-1]
    sub = levels.loc[(levels.index >= s) & (levels.index <= e)]
    return sub / sub.iloc[0] * 100.0


def window_stats(levels: pd.DataFrame, cols: dict[str, str], start, end=None, rf: float = 0.0) -> pd.DataFrame:
    """Standard statistics for several level columns over [start, end]."""
    sub = window(levels, start, end)
    rows = []
    for col, label in cols.items():
        lv = sub[col]
        r = M.daily_returns(lv)
        rows.append({
            "Series": label, "Start": lv.index[0].date(), "End": lv.index[-1].date(),
            "Total_Return": M.total_return(lv), "CAGR": M.cagr(lv), "Volatility": M.ann_vol(lv),
            "Sharpe": M.sharpe(lv, rf), "Sortino": M.sortino(lv, rf), "Max_Drawdown": M.max_drawdown(lv)["max_drawdown"],
            "Calmar": M.calmar(lv), "Skew_daily": float(r.skew()), "Kurtosis_daily": float(r.kurt()),
        })
    return pd.DataFrame(rows)


def hsmo_style_stats(level: pd.Series, rf: float = 0.02) -> dict:
    """Metrics computed exactly like `analyze_performance` in HSI300-Momentum-Strategy
    (CAGR from calendar of trading days incl. the first, Sharpe with a 2% risk-free rate)."""
    ret = level.pct_change().dropna()
    n = len(level)
    total = level.iloc[-1] / level.iloc[0] - 1
    cagr = (level.iloc[-1] / level.iloc[0]) ** (252 / n) - 1
    vol = ret.std() * np.sqrt(252)
    return {"Total_Return": float(total), "CAGR": float(cagr), "Volatility": float(vol),
            "Sharpe(rf=2%)": float((cagr - rf) / vol) if vol > 0 else np.nan,
            "Max_Drawdown": float(((level - level.cummax()) / level.cummax()).min())}


def monthly_distribution(strategy: pd.Series, benchmark: pd.Series) -> dict:
    ms = M.monthly_returns(strategy)
    mb = M.monthly_returns(benchmark).reindex(ms.index)
    ex = ms - mb
    up = mb > 0
    out = {
        "n_months": int(len(ms)),
        "strategy_mean": float(ms.mean()), "benchmark_mean": float(mb.mean()),
        "strategy_std": float(ms.std()), "benchmark_std": float(mb.std()),
        "strategy_skew": float(ms.skew()), "benchmark_skew": float(mb.skew()),
        "strategy_kurt": float(ms.kurt()), "benchmark_kurt": float(mb.kurt()),
        "strategy_pos_share": float((ms > 0).mean()), "benchmark_pos_share": float((mb > 0).mean()),
        "excess_mean": float(ex.mean()), "excess_std": float(ex.std()), "excess_pos_share": float((ex > 0).mean()),
        "excess_t_stat": float(ex.mean() / ex.std() * np.sqrt(len(ex))),
        "best_month_strategy": (ms.idxmax(), float(ms.max())), "worst_month_strategy": (ms.idxmin(), float(ms.min())),
        "best_month_benchmark": (mb.idxmax(), float(mb.max())), "worst_month_benchmark": (mb.idxmin(), float(mb.min())),
        "best_excess_month": (ex.idxmax(), float(ex.max())), "worst_excess_month": (ex.idxmin(), float(ex.min())),
        "up_capture": float(ms[up].mean() / mb[up].mean()), "down_capture": float(ms[~up].mean() / mb[~up].mean()),
        "hit_rate_up_months": float((ex[up] > 0).mean()), "hit_rate_down_months": float((ex[~up] > 0).mean()),
        "excess_in_up_months": float(ex[up].mean()), "excess_in_down_months": float(ex[~up].mean()),
    }
    # conditional betas (daily)
    rs, rb = M.daily_returns(strategy), M.daily_returns(benchmark).reindex(strategy.index[1:]).fillna(0)
    rs = rs.reindex(rb.index)
    upd = rb > 0
    out["beta_up_days"] = float(np.cov(rs[upd], rb[upd])[0, 1] / rb[upd].var())
    out["beta_down_days"] = float(np.cov(rs[~upd], rb[~upd])[0, 1] / rb[~upd].var())
    out["monthly_excess"] = ex
    return out


def rolling_beta_te(strategy: pd.Series, benchmark: pd.Series, win: int = TD) -> pd.DataFrame:
    rs = M.daily_returns(strategy)
    rb = M.daily_returns(benchmark).reindex(rs.index)
    cov = rs.rolling(win).cov(rb)
    var = rb.rolling(win).var()
    beta = cov / var
    te = (rs - rb).rolling(win).std() * np.sqrt(TD)
    ex = ((1 + rs).rolling(win).apply(np.prod, raw=True) - (1 + rb).rolling(win).apply(np.prod, raw=True))
    ir = ((rs - rb).rolling(win).mean() * TD) / te
    return pd.DataFrame({"beta": beta, "tracking_error": te, "rolling_excess_1y": ex, "information_ratio": ir}).dropna()


def tail_stats(strategy: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
    rows = []
    for name, lv in (("CSI 300", benchmark), ("Strategy", strategy)):
        r = M.daily_returns(lv)
        rows.append({"Series": name, "Daily_mean": r.mean(), "Daily_std": r.std(), "Skew": r.skew(), "Kurtosis": r.kurt(),
                     "VaR_95": r.quantile(0.05), "CVaR_95": r[r <= r.quantile(0.05)].mean(),
                     "VaR_99": r.quantile(0.01), "CVaR_99": r[r <= r.quantile(0.01)].mean(),
                     "Worst_day": r.min(), "Worst_day_date": r.idxmin().date(), "Best_day": r.max(),
                     "Days_below_-5%": int((r < -0.05).sum()), "Days_above_+5%": int((r > 0.05).sum())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 2. portfolio analytics from the per-date files
# --------------------------------------------------------------------------- #
def _holdings_files(cfg: dict) -> list[Path]:
    d = abs_path(cfg, cfg["output"]["holdings_dir"])
    return sorted(p for p in d.glob("*.csv") if "pending" not in p.name)


def _score_files(cfg: dict) -> list[Path]:
    return sorted(abs_path(cfg, cfg["output"]["scores_dir"]).glob("*.csv"))


def holdings_characteristics(cfg: dict) -> pd.DataFrame:
    rows = []
    for p in _holdings_files(cfg):
        h = pd.read_csv(p, dtype={"Ticker": str})
        w = h["Final_Weight"].values
        bw = h["CSI300_Weight"].values
        rank_by_size = pd.read_csv(abs_path(cfg, cfg["output"]["scores_dir"]) / f"{h['Reference_Date'].iloc[0]}.csv", dtype={"ticker": str})
        rank_by_size = rank_by_size[rank_by_size["csi300_member"]].sort_values("csi300_weight", ascending=False)
        top100 = set(rank_by_size["ticker"].head(100))
        rows.append({
            "reference_date": pd.Timestamp(h["Reference_Date"].iloc[0]), "effective_date": pd.Timestamp(h["Effective_Date"].iloc[0]),
            "n": len(h), "max_weight": w.max(), "top10_weight": np.sort(w)[::-1][:10].sum(),
            "effective_n": 1.0 / np.sum(w ** 2), "n_capped": int(h["Capped"].sum()) if "Capped" in h else np.nan,
            "n_buffer": int(h["Buffer_Retained"].sum()) if "Buffer_Retained" in h else np.nan,
            "csi300_weight_covered": bw.sum(),                 # share of CSI 300 (proxy) cap held
            "avg_active_weight_ratio": float(np.sum(w) / np.sum(bw)),   # >1: strategy overweights its names vs parent
            "weight_in_top100_by_size": float(w[h["Ticker"].isin(top100).values].sum()),
            "mean_momentum_score": float(h["Momentum_Score"].mean()),
            "min_momentum_score": float(h["Momentum_Score"].min()),
        })
    return pd.DataFrame(rows)


def persistence(cfg: dict, names: Optional[dict] = None) -> dict:
    files = _holdings_files(cfg)
    sets = []
    name_map = {}
    for p in files:
        h = pd.read_csv(p, dtype={"Ticker": str})
        sets.append(set(h["Ticker"]))
        name_map.update(dict(zip(h["Ticker"], h["Name"].fillna(""))))
    if names:
        name_map.update({k: v for k, v in names.items() if k not in name_map or not name_map[k]})
    counts = defaultdict(int)
    spells = []
    cur_spell: dict[str, int] = {}
    for i, s in enumerate(sets):
        for c in s:
            counts[c] += 1
            cur_spell[c] = cur_spell.get(c, 0) + 1
        for c in list(cur_spell):
            if c not in s:
                spells.append(cur_spell.pop(c))
    spells += list(cur_spell.values())
    retained = [len(sets[i] & sets[i - 1]) / len(sets[i - 1]) for i in range(1, len(sets))]
    cnt = pd.Series(counts).sort_values(ascending=False)
    top = pd.DataFrame({"ticker": cnt.index, "name": [name_map.get(c, "") for c in cnt.index], "periods_selected": cnt.values})
    return {"n_periods": len(sets), "n_distinct": len(cnt), "counts": cnt, "top": top,
            "spells": pd.Series(spells), "mean_spell": float(np.mean(spells)), "median_spell": float(np.median(spells)),
            "share_single_period": float((pd.Series(spells) == 1).mean()),
            "avg_retention": float(np.mean(retained)), "retention": pd.Series(retained, index=[pd.Timestamp(pd.read_csv(p, nrows=1)["Effective_Date"].iloc[0]) for p in files[1:]])}


def fv_funnel(cfg: dict) -> pd.DataFrame:
    rows = []
    for p in _score_files(cfg):
        s = pd.read_csv(p, dtype={"ticker": str})
        s = s[s["csi300_member"]]
        reasons = s.loc[~s["financial_eligible"], "fin_reason"].fillna("unknown").astype(str)
        cat = reasons.str.extract(r"^([^:(]+)")[0].str.strip()
        vc = cat.value_counts()
        rows.append({"reference_date": pd.Timestamp(s["reference_date"].iloc[0]), "members": len(s),
                     "financial_eligible": int(s["financial_eligible"].sum()),
                     "momentum_eligible": int(s["momentum_eligible"].sum()), "eligible": int(s["eligible"].sum()),
                     "selected": int(s["selected"].sum()),
                     "z_clipped": int((s.loc[s["eligible"], "z_score"].abs() >= 3 - 1e-9).sum()),
                     "score_max": float(s.loc[s["eligible"], "momentum_score"].max()),
                     "score_median": float(s.loc[s["eligible"], "momentum_score"].median()),
                     "score_cutoff": float(s.loc[s["selected"], "momentum_score"].min()) if s["selected"].any() else np.nan,
                     **{f"fail:{k}": int(v) for k, v in vc.items()}})
    df = pd.DataFrame(rows).fillna(0)
    return df


def sector_exposure(cfg: dict, sector_map: pd.DataFrame, official_weights: pd.DataFrame) -> pd.DataFrame:
    """Latest implemented holdings and the pending list vs the official CSI 300 weights, by CSI sector."""
    smap = dict(zip(sector_map["code"], sector_map["sector"]))
    hold_dir = abs_path(cfg, cfg["output"]["holdings_dir"])
    files = sorted(hold_dir.glob("*.csv"))
    out = {}
    off = official_weights.assign(sector=official_weights["code"].map(smap).fillna("未分类"))
    out[f"CSI 300 ({official_weights['as_of'].iloc[0].date()})"] = off.groupby("sector")["weight"].sum()
    for p in files[-2:]:
        h = pd.read_csv(p, dtype={"Ticker": str})
        h["sector"] = h["Ticker"].map(smap).fillna("未分类")
        label = ("Strategy " + p.stem.replace("_pending", " (pending)"))
        out[label] = h.groupby("sector")["Final_Weight"].sum()
    df = pd.DataFrame(out).fillna(0.0)
    return df.sort_values(df.columns[0], ascending=False)


# --------------------------------------------------------------------------- #
# 3. attribution runs with the index engine
# --------------------------------------------------------------------------- #
def _rebalances_from_holdings(cfg: dict, scheme: str, panel: dict, intervals: pd.DataFrame) -> list[Rebalance]:
    rebs = []
    for p in _holdings_files(cfg):
        h = pd.read_csv(p, dtype={"Ticker": str})
        ref = pd.Timestamp(h["Reference_Date"].iloc[0]); eff = pd.Timestamp(h["Effective_Date"].iloc[0])
        if scheme == "capweight":
            w = h.set_index("Ticker")["Float_Market_Cap"]
        elif scheme == "equal":
            w = pd.Series(1.0, index=h["Ticker"])
        else:
            raise ValueError(scheme)
        w = w / w.sum()
        rebs.append(Rebalance(ref, eff, w, {"scheme": scheme}))
    return rebs


def _replica_rebalances(cfg: dict, panel: dict, intervals: pd.DataFrame) -> list[Rebalance]:
    """All CSI 300 members on each implementation date, weighted by the project's float-cap proxy."""
    rebs = []
    fmc = panel["fmc"]
    for p in _holdings_files(cfg):
        h = pd.read_csv(p, dtype={"Ticker": str}, nrows=1)
        ref = pd.Timestamp(h["Reference_Date"].iloc[0]); eff = pd.Timestamp(h["Effective_Date"].iloc[0])
        members = sorted(members_at(intervals, eff))
        w = fmc.loc[ref].reindex(members).dropna()
        w = w[w > 0]
        rebs.append(Rebalance(ref, eff, w / w.sum(), {"scheme": "replica"}))
    return rebs


def attribution_runs(cfg: dict, cal: TradingCalendar, intervals: pd.DataFrame, panel: dict) -> pd.DataFrame:
    engine = IndexEngine(panel, cal, intervals, cfg)
    base = 100.0
    out = {}
    for label, rebs in (("Selected_basket_capweighted", _rebalances_from_holdings(cfg, "capweight", panel, intervals)),
                        ("Selected_basket_equalweighted", _rebalances_from_holdings(cfg, "equal", panel, intervals)),
                        ("CSI300_replica_proxy", _replica_rebalances(cfg, panel, intervals))):
        res = engine.run(rebs, basis="pr", base_level=base)
        out[label] = res["levels"]["Close"]
        log.info("attribution run %s: final %.1f", label, out[label].iloc[-1])
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# 4. HSI300-Momentum-Strategy re-implementation on this project's data
# --------------------------------------------------------------------------- #
def hsmo_replication(panel: dict, intervals: pd.DataFrame, start, end=None, top_n: int = 20, cost_rate: float = 0.0015,
                     max_per_ind: Optional[int] = None, sector_map: Optional[dict] = None, price_field: str = "adj_close_tr",
                     cost_cfg: Optional[dict] = None) -> dict:
    """Faithful port of `MomentumStrategy.run` in mo_hsi300_exact_drift.py.

    * score_t = [P(t-22)/P(t-253) - 1] / std(daily returns over the 252 days ending t-1]  (their shift(1))
    * rebalance on the last trading day of each calendar quarter (pandas '3ME') and on the first day
    * universe = CSI 300 members on the rebalance day (this project's daily point-in-time membership
      instead of their monthly BaoStock snapshots)
    * top_n names, optional max_per_ind per (current) sector, equal weight, natural drift in between
    * cost = cost_rate x traded value (both directions); with `cost_cfg` (this project's cost schedule)
      buys and sells are charged the historical A-share commission / stamp duty / transfer / slippage rates
    Prices: this project's total-return adjusted close (their BaoStock 后复权), forward-filled.
    """
    P = panel[price_field].ffill()
    mom = P.shift(21) / P.shift(252) - 1
    vol = P.pct_change().rolling(252).std()
    score = (mom / vol).shift(1)
    dates = P.index[P.index >= pd.Timestamp(start)]
    if end is not None:
        dates = dates[dates <= pd.Timestamp(end)]
    dates = dates[dates >= P.index[253]]
    rb = set(P.index.to_series().resample("3ME").last().values)
    cur = pd.Series(0.0, index=P.columns)
    capital = 100.0
    curve, turnover = [], []
    holdings = {}
    for t, d in enumerate(dates):
        if t > 0:
            r = (P.loc[d] / P.loc[dates[t - 1]] - 1).fillna(0.0)
            cur = cur * (1 + r)
        total = capital if t == 0 else cur.sum()
        if d in rb or t == 0:
            uni = members_at(intervals, d)
            sc = score.loc[d].reindex(sorted(uni)).dropna().sort_values(ascending=False)
            sel = []
            cnt = defaultdict(int)
            for c in sc.index:
                ind = (sector_map or {}).get(c, "Unknown") if max_per_ind else None
                if max_per_ind is None or cnt[ind] < max_per_ind:
                    sel.append(c)
                    if max_per_ind:
                        cnt[ind] += 1
                if len(sel) >= top_n:
                    break
            target = pd.Series(0.0, index=P.columns)
            if sel:
                target[sel] = total / len(sel)
            diff = target - cur
            traded = diff.abs().sum()
            if cost_cfg is not None:
                buy_rate, sell_rate = cost_rates(cost_cfg, d)
                cost = diff.clip(lower=0).sum() * buy_rate + (-diff.clip(upper=0)).sum() * sell_rate
            else:
                cost = traded * cost_rate
            net = total - cost
            cur[:] = 0.0
            if sel:
                cur[sel] = net / len(sel)
            turnover.append((d, traded / total if total > 0 else np.nan, cost))
            holdings[d] = list(sel)
        curve.append(cur.sum())
    lv = pd.Series(curve, index=dates, name="HSMO_replication")
    to = pd.DataFrame(turnover, columns=["date", "two_way_turnover", "cost"]).set_index("date")
    return {"levels": lv, "turnover": to, "holdings": holdings, "n_rebalances": len(to)}


# --------------------------------------------------------------------------- #
# 5. design-space sensitivity: one axis at a time between this index and a
#    concentrated top-N equal-weight portfolio (HSMO-style).
#    DIAGNOSTIC ONLY — the published index keeps the fixed rules of config.yaml.
# --------------------------------------------------------------------------- #
QUARTERLY = ((2, 3), (5, 6), (8, 9), (11, 12))
SEMIANNUAL = ((2, 3), (8, 9))


def variant_schedule(cal: TradingCalendar, cfg: dict, data_end: pd.Timestamp, months=SEMIANNUAL,
                     lag: str = "third_friday") -> list[tuple]:
    """(reference, implementation) pairs. `lag`: "third_friday" = the index convention (month-end
    reference, third Friday of the following month, ~3 weeks of pre-announcement); "immediate" =
    trade at the reference-date close (what an unannounced portfolio can do)."""
    first_ref = pd.Timestamp(cfg["rebalance"]["first_reference_date"])
    out = []
    for y in range(first_ref.year, data_end.year + 1):
        for m_ref, m_eff in months:
            ref = cal.month_end(y, m_ref)
            if ref < first_ref or ref > data_end:
                continue
            yy = y + 1 if m_eff < m_ref else y          # December reference -> January implementation
            out.append((ref, ref if lag == "immediate" else cal.third_friday(yy, m_eff)))
    return sorted(out)


def score_cache(cfg: dict, cal: TradingCalendar, intervals: pd.DataFrame, panel: dict, qtabs: dict,
                refs: Iterable[pd.Timestamp]) -> dict:
    """Per-reference-date eligibility / momentum / market-cap frame, shared by all design variants
    (identical to the production scoring step; only selection and weighting differ downstream)."""
    from .financials import viability_table
    from .momentum import compute_momentum
    from .universe import members_at as _members_at

    fv, mcfg = cfg["financial_viability"], cfg["momentum"]
    cache = {}
    for ref in refs:
        members = sorted(_members_at(intervals, ref))
        fin = viability_table(qtabs, members, ref, require_latest_positive=fv["require_latest_quarter_positive"],
                              require_4q_positive=fv["require_trailing_4q_sum_positive"])
        mom = compute_momentum(panel["adj_close_pr"], panel["ret_pr"], panel["traded"], ref, cal, codes=members,
                               lookback_months=mcfg["lookback_months"], skip_months=mcfg["skip_months"],
                               min_traded_days=mcfg["min_traded_days"])
        mom6 = compute_momentum(panel["adj_close_pr"], panel["ret_pr"], panel["traded"], ref, cal, codes=members,
                                lookback_months=6, skip_months=mcfg["skip_months"], min_traded_days=90)
        mom = mom.merge(mom6[["code", "RAM"]].rename(columns={"RAM": "RAM_6"}), on="code", how="left")
        df = pd.DataFrame({"code": members}).merge(fin, on="code", how="left").merge(mom, on="code", how="left")
        df["momentum_eligible"] = df["momentum_eligible"].fillna(False).astype(bool)
        df["financial_eligible"] = df["financial_eligible"].fillna(False).astype(bool)
        fmc = panel["fmc"].loc[ref].reindex(members)
        df["fmc"] = fmc.values
        df["csi300_weight"] = (fmc / fmc.dropna().sum()).values
        cache[ref] = df
    return cache


def variant_rebalances(cfg: dict, cache: dict, intervals: pd.DataFrame, schedule: list[tuple], data_end: pd.Timestamp,
                       top_fraction: float | None = 0.20, top_n: int | None = None, weighting: str = "fmc_score_cap",
                       use_fv: bool = True, buffer: bool = True, signal: str = "12-1") -> list:
    """Build Rebalance objects for one design variant. Parent membership always has priority."""
    from .selection import select_constituents
    from .universe import members_at as _members_at
    from .weighting import compute_weights

    scfg, wcfg = cfg["selection"], cfg["weighting"]
    auto_frac = scfg["buffer"]["auto_select_fraction"] if buffer else 1.0
    retain_frac = scfg["buffer"]["retain_fraction"] if buffer else 1.0
    rebs, current = [], set()
    for ref, eff in schedule:
        df = cache[ref].copy()
        df["eligible"] = df["momentum_eligible"] & df["fmc"].notna() & (df["fmc"] > 0)
        if use_fv:
            df["eligible"] &= df["financial_eligible"]
        if signal == "blend":      # average of the standardised 12-1 and 6-1 signals (re-standardised downstream)
            from .selection import zscore

            m = df["eligible"] & df["RAM_6"].notna()
            df.loc[~m, "eligible"] = False
            if int(m.sum()) >= 5:
                df.loc[m, "RAM"] = 0.5 * (zscore(df.loc[m, "RAM"].astype(float)) + zscore(df.loc[m, "RAM_6"].astype(float))).values
        n_elig = int(df["eligible"].sum())
        if n_elig < 5:
            continue
        frac = (top_n / n_elig) if top_n else top_fraction
        sel = select_constituents(df, current, top_fraction=frac, auto_frac=auto_frac, retain_frac=retain_frac,
                                  rounding=scfg["rounding"])
        chosen = sel[sel["selected"]].copy()
        if weighting == "equal":
            chosen["final_weight"] = 1.0 / len(chosen)
        else:
            w_in = chosen.copy()
            if weighting == "fmc_cap":
                w_in["momentum_score"] = 1.0
            elif weighting == "sqrt_fmc_score_cap":
                w_in["fmc"] = np.sqrt(w_in["fmc"].astype(float))
            elif weighting != "fmc_score_cap":
                raise ValueError(weighting)
            chosen["final_weight"] = compute_weights(w_in, cap_absolute=wcfg["cap_absolute"],
                                                     cap_multiple=wcfg["cap_multiple_of_parent_weight"],
                                                     max_iter=wcfg["max_cap_iterations"])["final_weight"].values
        mem_eff = _members_at(intervals, eff) if eff <= data_end else set(df["code"])
        keep = chosen[chosen["code"].isin(mem_eff)].copy()
        if keep.empty:
            continue
        w = pd.Series(keep["final_weight"].values, index=keep["code"].values)
        w = w / w.sum()
        current = set(w.index)
        rebs.append(Rebalance(ref, eff, w, {"n": len(w)}))
    return rebs


def run_variant(cfg: dict, engine: IndexEngine, rebs: list, basis: str = "tr", base: float = 100.0) -> dict:
    res = engine.run(rebs, basis=basis, base_level=base)
    lev = res["levels"]
    to = res["turnover"]["one_way_turnover"].iloc[1:]
    years = M.years_between(lev.index[0], lev.index[-1])
    n_per_year = len(to) / years
    ns = [len(r.weights) for r in rebs]
    eff_n = [1.0 / float((r.weights ** 2).sum()) for r in rebs]
    return {"levels": lev["Close"], "levels_net": lev["Net_Close"], "turnover": to,
            "stats": {"CAGR_gross": M.cagr(lev["Close"]), "CAGR_net": M.cagr(lev["Net_Close"]),
                      "Vol": M.ann_vol(lev["Net_Close"]), "Sharpe": M.sharpe(lev["Net_Close"]),
                      "MaxDD": M.max_drawdown(lev["Net_Close"])["max_drawdown"],
                      "Turnover_pa": float(to.mean() * n_per_year), "N": float(np.mean(ns)), "EffN": float(np.mean(eff_n)),
                      "Rebalances": len(rebs), "Final_net": float(lev["Net_Close"].iloc[-1])}}


def design_grid(cfg: dict, cal: TradingCalendar, intervals: pd.DataFrame, panel: dict, qtabs: dict,
                variants: list[dict], split: str = "2016-01-01") -> tuple[pd.DataFrame, dict]:
    """Run a list of design variants on identical point-in-time data. Each variant dict:
    {name, top_fraction|top_n, freq: 'semiannual'|'quarterly', weighting, use_fv, buffer}."""
    data_end = panel["adj_close_pr"].index[-1]
    def months_for(freq: str, phase: int) -> tuple:
        step = 6 if freq == "semiannual" else 3
        return tuple(((phase + k * step - 1) % 12 + 1, (phase + k * step) % 12 + 1) for k in range(12 // step))

    scheds = {}
    for v in variants:
        key = (v.get("freq", "semiannual"), v.get("lag", "third_friday"), v.get("phase", 2))
        if key not in scheds:
            scheds[key] = variant_schedule(cal, cfg, data_end, months_for(key[0], key[2]), key[1])
    refs = sorted({r for sch in scheds.values() for r, _ in sch})
    cache = score_cache(cfg, cal, intervals, panel, qtabs, refs)
    # engine on a forward-filled copy: some variants can hold a security that is suspended until it is
    # delisted (no price at all); the index rule carries such holdings at their last close
    panel_ff = dict(panel)
    for k in ("adj_close_pr", "adj_open_pr", "adj_high_pr", "adj_low_pr", "adj_close_tr", "close"):
        panel_ff[k] = panel[k].ffill()
    engine = IndexEngine(panel_ff, cal, intervals, cfg)
    rows, curves, raw = [], {}, {}
    for v in variants:
        rebs = variant_rebalances(cfg, cache, intervals, scheds[(v.get("freq", "semiannual"), v.get("lag", "third_friday"), v.get("phase", 2))], data_end,
                                  top_fraction=v.get("top_fraction", 0.20), top_n=v.get("top_n"),
                                  weighting=v.get("weighting", "fmc_score_cap"), use_fv=v.get("use_fv", True),
                                  buffer=v.get("buffer", True), signal=v.get("signal", "12-1"))
        raw[v["name"]] = run_variant(cfg, engine, rebs)
    # compare on the window common to every variant (an immediate-implementation variant starts
    # ~3 weeks earlier than the index convention)
    t_start = max(o["levels_net"].index[0] for o in raw.values())
    for name, out in raw.items():
        lv = out["levels_net"].loc[t_start:]
        lv = lv / lv.iloc[0] * 100.0
        curves[name] = lv
        st = out["stats"]
        s = {"variant": name, "group": next(v.get("group", "") for v in variants if v["name"] == name), "CAGR_net": M.cagr(lv), "CAGR_gross": M.cagr(out["levels"].loc[t_start:]),
             "Vol": M.ann_vol(lv), "Sharpe": M.sharpe(lv), "MaxDD": M.max_drawdown(lv)["max_drawdown"],
             "Turnover_pa": st["Turnover_pa"], "N": st["N"], "EffN": st["EffN"], "Rebalances": st["Rebalances"],
             "Final_net": float(lv.iloc[-1])}
        for label, sub in (("first_half", lv.loc[:split]), ("second_half", lv.loc[split:])):
            s[f"CAGR_net_{label}"] = M.cagr(sub)
        rows.append(s)
        log.info("variant %-46s CAGR_net %.2f%% vol %.1f%% MDD %.0f%% turnover %.0f%%/yr N %.0f",
                 name, 100 * s["CAGR_net"], 100 * s["Vol"], 100 * s["MaxDD"], 100 * s["Turnover_pa"], s["N"])
    return pd.DataFrame(rows), curves


def tranche_curve(curves: dict, keys) -> pd.Series:
    """Equal-capital combination of sub-portfolios that follow the same rules on staggered
    rebalance months (Jegadeesh-Titman overlapping portfolios). Turnover per unit of capital is
    unchanged; only the arbitrary choice of rebalance month is diversified away."""
    rets = pd.DataFrame({k: curves[k].pct_change() for k in keys}).dropna()
    lv = (1 + rets.mean(axis=1)).cumprod() * 100.0
    lv.loc[rets.index[0] - pd.Timedelta(days=1)] = 100.0
    return lv.sort_index()


def tranching_dispersion(curves: dict, keys, ks=(1, 2, 3, 6), split: str = "2016-01-01") -> pd.DataFrame:
    """Outcome dispersion when k of the staggered sub-portfolios are combined (all k-subsets)."""
    from itertools import combinations

    keys = list(keys)
    rets = pd.DataFrame({k: curves[k].pct_change() for k in keys}).dropna()
    rows = []
    for k in ks:
        stats = []
        for cmb in combinations(keys, k):
            lv = (1 + rets[list(cmb)].mean(axis=1)).cumprod() * 100.0
            stats.append((M.cagr(lv), M.ann_vol(lv), M.sharpe(lv), M.max_drawdown(lv)["max_drawdown"]))
        cg = np.array([s[0] for s in stats])
        rows.append({"分批数": k, "组合数": len(stats), "CAGR 均值": cg.mean(), "CAGR 最低": cg.min(), "CAGR 最高": cg.max(),
                     "极差(pp)": 100 * (cg.max() - cg.min()), "标准差(pp)": 100 * cg.std(),
                     "波动": np.mean([s[1] for s in stats]), "Sharpe": np.mean([s[2] for s in stats]),
                     "最大回撤": np.mean([s[3] for s in stats])})
    return pd.DataFrame(rows)


DESIGN_VARIANTS = [
    {"group": "基准", "name": "A 本指数：top20%，半年，FMC×Score+上限，FV，缓冲，三周后实施"},
    {"group": "成分数", "name": "B1 成分数 top10%", "top_fraction": 0.10},
    {"group": "成分数", "name": "B2 成分数 top5%", "top_fraction": 0.05},
    {"group": "成分数", "name": "B3 成分数 固定20只", "top_n": 20},
    {"group": "频率", "name": "C 频率 季度", "freq": "quarterly"},
    {"group": "加权", "name": "D1 加权 等权", "weighting": "equal"},
    {"group": "加权", "name": "D2 加权 仅FMC", "weighting": "fmc_cap"},
    {"group": "加权", "name": "D3 加权 √FMC×Score", "weighting": "sqrt_fmc_score_cap"},
    {"group": "筛选", "name": "E 无财务资格筛选", "use_fv": False},
    {"group": "筛选", "name": "F 无缓冲", "buffer": False},
    {"group": "实施时点", "name": "L 参考日收盘实施（无三周延迟）", "lag": "immediate"},
    {"group": "折中", "name": "G1 top10%+季度", "top_fraction": 0.10, "freq": "quarterly"},
    {"group": "折中", "name": "G2 季度+参考日实施", "freq": "quarterly", "lag": "immediate"},
    {"group": "折中", "name": "G3 top10%+季度+参考日实施", "top_fraction": 0.10, "freq": "quarterly", "lag": "immediate"},
    {"group": "折中", "name": "G4 top10%+季度+参考日实施+√FMC×Score", "top_fraction": 0.10, "freq": "quarterly", "lag": "immediate",
     "weighting": "sqrt_fmc_score_cap"},
    {"group": "HSMO 端点", "name": "H1 20只+季度+等权+无FV/缓冲", "top_n": 20, "freq": "quarterly", "weighting": "equal",
     "use_fv": False, "buffer": False},
    {"group": "HSMO 端点", "name": "H2 H1 + 参考日实施", "top_n": 20, "freq": "quarterly", "weighting": "equal",
     "use_fv": False, "buffer": False, "lag": "immediate"},
]
PHASE_VARIANTS = [{"group": "日历相位", "name": f"T{p} 本指数规则，参考月 {p}/{p + 6 if p <= 6 else p - 6}", "phase": p} for p in range(1, 7)]


def design_study(cfg: dict, cal: TradingCalendar, intervals: pd.DataFrame, panel: dict, qtabs: dict,
                 benchmark: pd.Series) -> dict:
    """Design-space sensitivity + calendar-phase robustness + tranching (diagnostic only)."""
    gr, curves = design_grid(cfg, cal, intervals, panel, qtabs, DESIGN_VARIANTS + PHASE_VARIANTS)
    phase_names = [v["name"] for v in PHASE_VARIANTS]
    grid = gr[~gr["variant"].isin(phase_names)].reset_index(drop=True)
    phase = gr[gr["variant"].isin(phase_names)].reset_index(drop=True)
    disp = tranching_dispersion(curves, phase_names)
    tr = tranche_curve(curves, phase_names)
    bench = benchmark.reindex(tr.index).ffill().dropna()
    bench = bench / bench.iloc[0] * 100
    published = curves[phase_names[1]]           # T2 = the published Feb/Aug phase
    summary = {
        "bench_cagr": M.cagr(bench), "published_cagr": M.cagr(published), "phase_mean": float(phase["CAGR_net"].mean()),
        "phase_min": float(phase["CAGR_net"].min()), "phase_max": float(phase["CAGR_net"].max()),
        "phase_std": float(phase["CAGR_net"].std()), "tranched_cagr": M.cagr(tr), "tranched_vol": M.ann_vol(tr),
        "tranched_sharpe": M.sharpe(tr), "tranched_mdd": M.max_drawdown(tr)["max_drawdown"],
        "tranched_turnover": float(phase["Turnover_pa"].mean()),
        "design_min": float(grid["CAGR_net"].min()), "design_max": float(grid["CAGR_net"].max()),
    }
    # --- is any of this distinguishable from noise, and is there something better? -------------
    A_curve = curves[DESIGN_VARIANTS[0]["name"]]
    burn = A_curve.pct_change().dropna().index[504]          # 2y burn-in for an out-of-sample vol target
    vt = vol_scaled(A_curve, target_vol="expanding", window=126, max_exposure=1.0)
    over = []
    for tv_, lab in (("expanding", "扩展窗口（无前视）"), (None, "全样本波动（含前视）"), (0.25, "固定 25%")):
        for win in (63, 126, 252):
            o = vol_scaled(A_curve, target_vol=tv_, window=win, max_exposure=1.0)
            x = o["levels"].loc[burn:]
            x = x / x.iloc[0] * 100
            over.append({"目标波动": lab, "估计窗口": win, "CAGR": M.cagr(x), "波动": M.ann_vol(x), "Sharpe": M.sharpe(x),
                         "最大回撤": M.max_drawdown(x)["max_drawdown"], "平均仓位": float(o["exposure"].loc[burn:].mean()),
                         "叠加换手/年": o["stats"]["overlay_turnover_pa"]})
    base_clip = A_curve.loc[burn:] / A_curve.loc[burn:].iloc[0] * 100
    overlay = pd.DataFrame([{"目标波动": "无（基准 A）", "估计窗口": np.nan, "CAGR": M.cagr(base_clip), "波动": M.ann_vol(base_clip),
                             "Sharpe": M.sharpe(base_clip), "最大回撤": M.max_drawdown(base_clip)["max_drawdown"],
                             "平均仓位": 1.0, "叠加换手/年": 0.0}] + over)
    pairs = [("A 本指数 − 沪深300全收益", A_curve, benchmark.reindex(A_curve.index).ffill().dropna(), None),
             ("H2 HSMO 端点（参考日实施） − A", curves["H2 H1 + 参考日实施"], A_curve, None),
             ("H1 HSMO 端点（三周后实施） − A", curves["H1 20只+季度+等权+无FV/缓冲"], A_curve, None),
             ("B1 成分数 top10% − A", curves["B1 成分数 top10%"], A_curve, None),
             ("C 季度调仓 − A", curves["C 频率 季度"], A_curve, None),
             ("D1 等权 − A", curves["D1 加权 等权"], A_curve, None),
             ("E 去掉财务资格筛选 − A", curves["E 无财务资格筛选"], A_curve, None),
             ("分批 6 批 − A", tr, A_curve, None),
             ("波动率目标（扩展窗口，126 日） − A", vt["levels"], A_curve, burn)]
    sig = []
    for lab, a_, b_, st_ in pairs:
        a2, b2 = (a_.loc[st_:], b_.loc[st_:]) if st_ is not None else (a_, b_)
        r_ = block_bootstrap_diff(a2, b2)
        sig.append({"比较": lab, "年化差(pp)": 100 * r_["ann_diff"], "区间下(pp)": 100 * r_["ci_low"],
                    "区间上(pp)": 100 * r_["ci_high"], "t(NW)": r_["t_newey_west"], "P(差>0)": r_["p_positive"],
                    "月数": r_["n_months"]})
    summary["te"] = M.tracking_error(A_curve, benchmark.reindex(A_curve.index).ffill().dropna())
    summary["se_excess"] = summary["te"] / np.sqrt(M.years_between(A_curve.index[0], A_curve.index[-1]))
    summary["burn_start"] = str(burn.date())
    return {"grid": grid, "phase": phase, "dispersion": disp, "tranched": tr, "curves": curves, "summary": summary,
            "significance": pd.DataFrame(sig), "overlay": overlay, "vol_target": vt, "A_curve": A_curve, "burn": burn}


# --------------------------------------------------------------------------- #
# 6. how much of any of this is distinguishable from noise?
# --------------------------------------------------------------------------- #
def block_bootstrap_diff(a: pd.Series, b: pd.Series, block: int = 12, n_boot: int = 2000, seed: int = 0) -> dict:
    """Moving-block bootstrap on the monthly return difference of two level series.
    Returns the annualised mean difference, its 95% interval and a Newey-West t-statistic."""
    ra, rb = M.monthly_returns(a), M.monthly_returns(b)
    idx = ra.index.intersection(rb.index)
    d = (ra.reindex(idx) - rb.reindex(idx)).dropna().values
    n = len(d)
    rng = np.random.default_rng(seed)
    nblocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, nblocks))
    means = np.empty(n_boot)
    for i in range(n_boot):
        sample = np.concatenate([d[s:s + block] for s in starts[i]])[:n]
        means[i] = sample.mean()
    ann = lambda m: (1 + m) ** 12 - 1                                   # noqa: E731
    lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    dm = d - d.mean()
    gamma0 = float((dm ** 2).mean())
    var_nw = gamma0 + 2 * sum((1 - k / (lags + 1)) * float((dm[k:] * dm[:-k]).mean()) for k in range(1, lags + 1))
    t_nw = d.mean() / np.sqrt(max(var_nw, 1e-18) / n)
    return {"ann_diff": ann(d.mean()), "ci_low": ann(np.percentile(means, 2.5)), "ci_high": ann(np.percentile(means, 97.5)),
            "t_newey_west": float(t_nw), "n_months": n, "p_positive": float((means > 0).mean())}


def vol_scaled(level: pd.Series, target_vol: float | None = None, window: int = 126, max_exposure: float = 1.0,
               cash_rate: float = 0.02, cost_per_unit: float = 0.0015) -> dict:
    """Constant-volatility overlay (Barroso and Santa-Clara 2015): scale exposure by
    target_vol / realised vol of the strategy itself, estimated on the previous `window` days
    (strictly lagged). Un-invested capital earns `cash_rate`; changing exposure costs
    `cost_per_unit` of the traded fraction."""
    r = level.pct_change().dropna()
    rv = r.rolling(window).std().shift(1) * np.sqrt(252)
    if target_vol == "expanding":       # no look-ahead: target = realised vol of the strategy so far
        tv = (r.expanding(504).std() * np.sqrt(252)).shift(1)
    else:
        tv = float(r.std() * np.sqrt(252)) if target_vol is None else float(target_vol)
    e = (tv / rv).clip(upper=max_exposure).fillna(0.0)
    e = e.where(rv.notna(), 0.0)
    turn = e.diff().abs().fillna(e.abs())
    rp = e * r + (1 - e) * (cash_rate / 252) - turn * cost_per_unit
    lv = (1 + rp).cumprod() * 100.0
    lv.loc[r.index[0] - pd.Timedelta(days=1)] = 100.0
    lv = lv.sort_index()
    return {"levels": lv, "exposure": e, "target_vol": (float(np.nanmean(tv)) if not np.isscalar(tv) else tv),
            "stats": {"CAGR": M.cagr(lv), "Vol": M.ann_vol(lv), "Sharpe": M.sharpe(lv),
                      "MaxDD": M.max_drawdown(lv)["max_drawdown"], "avg_exposure": float(e.mean()),
                      "overlay_turnover_pa": float(turn.sum() / M.years_between(lv.index[0], lv.index[-1]))}}
