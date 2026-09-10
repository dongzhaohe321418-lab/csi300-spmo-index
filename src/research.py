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
from typing import Optional

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
