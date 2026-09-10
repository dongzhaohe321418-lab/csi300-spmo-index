"""Performance and risk statistics for daily index levels."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 244


def total_return(level: pd.Series) -> float:
    return float(level.iloc[-1] / level.iloc[0] - 1.0)


def years_between(a, b) -> float:
    return (pd.Timestamp(b) - pd.Timestamp(a)).days / 365.25


def cagr(level: pd.Series) -> float:
    yrs = years_between(level.index[0], level.index[-1])
    if yrs <= 0:
        return np.nan
    return float((level.iloc[-1] / level.iloc[0]) ** (1.0 / yrs) - 1.0)


def daily_returns(level: pd.Series) -> pd.Series:
    return level.pct_change().dropna()


def ann_vol(level: pd.Series, periods: int = TRADING_DAYS) -> float:
    return float(daily_returns(level).std(ddof=1) * np.sqrt(periods))


def sharpe(level: pd.Series, rf: float = 0.0, periods: int = TRADING_DAYS) -> float:
    r = daily_returns(level) - rf / periods
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(periods)) if sd > 0 else np.nan


def sortino(level: pd.Series, rf: float = 0.0, periods: int = TRADING_DAYS) -> float:
    r = daily_returns(level) - rf / periods
    downside = np.sqrt((np.minimum(r, 0.0) ** 2).mean())
    return float(r.mean() / downside * np.sqrt(periods)) if downside > 0 else np.nan


def drawdown_series(level: pd.Series) -> pd.Series:
    return level / level.cummax() - 1.0


def max_drawdown(level: pd.Series) -> dict:
    dd = drawdown_series(level)
    trough = dd.idxmin()
    peak = level.loc[:trough].idxmax()
    after = level.loc[trough:]
    rec = after[after >= level.loc[peak]]
    recovery = rec.index[0] if len(rec) else pd.NaT
    return {
        "max_drawdown": float(dd.min()),
        "peak_date": peak,
        "bottom_date": trough,
        "recovery_date": recovery,
        "days_to_bottom": int((trough - peak).days),
        "days_to_recovery": int((recovery - trough).days) if recovery is not pd.NaT else None,
    }


def calmar(level: pd.Series) -> float:
    mdd = max_drawdown(level)["max_drawdown"]
    return float(cagr(level) / abs(mdd)) if mdd < 0 else np.nan


def period_returns(level: pd.Series, freq: str) -> pd.Series:
    """Returns per calendar period (Y or M), first period measured from the start level."""
    lv = level.copy()
    ends = lv.groupby(lv.index.to_period(freq)).tail(1)
    starts = pd.concat([lv.iloc[[0]], ends.iloc[:-1]])
    out = pd.Series(ends.values / starts.values - 1.0, index=ends.index.to_period(freq))
    return out


def annual_returns(level: pd.Series) -> pd.Series:
    return period_returns(level, "Y")


def monthly_returns(level: pd.Series) -> pd.Series:
    return period_returns(level, "M")


def rolling_return(level: pd.Series, years: int, annualise: bool) -> pd.Series:
    """Return over the trailing `years` calendar years at every date (as-of lookup)."""
    idx = level.index
    start_dates = idx - pd.DateOffset(years=years)
    pos = idx.searchsorted(start_dates, side="right") - 1
    valid = (pos >= 0) & (start_dates >= idx[0])
    base = np.where(valid, level.values[np.clip(pos, 0, None)], np.nan)
    r = level.values / base - 1.0
    if annualise:
        r = (1.0 + r) ** (1.0 / years) - 1.0
    return pd.Series(r, index=idx).dropna()


def beta_alpha(strategy: pd.Series, benchmark: pd.Series, periods: int = TRADING_DAYS) -> tuple[float, float]:
    rs = daily_returns(strategy)
    rb = daily_returns(benchmark).reindex(rs.index)
    cov = np.cov(rs.values, rb.values, ddof=1)
    beta = cov[0, 1] / cov[1, 1]
    alpha_daily = rs.mean() - beta * rb.mean()
    return float(beta), float((1.0 + alpha_daily) ** periods - 1.0)


def tracking_error(strategy: pd.Series, benchmark: pd.Series, periods: int = TRADING_DAYS) -> float:
    rs = daily_returns(strategy)
    rb = daily_returns(benchmark).reindex(rs.index)
    return float((rs - rb).std(ddof=1) * np.sqrt(periods))


def information_ratio(strategy: pd.Series, benchmark: pd.Series, periods: int = TRADING_DAYS) -> float:
    te = tracking_error(strategy, benchmark, periods)
    return float((cagr(strategy) - cagr(benchmark)) / te) if te > 0 else np.nan


def summary(level: pd.Series, name: str, rf: float = 0.0, periods: int = TRADING_DAYS) -> dict:
    ann = annual_returns(level)
    mon = monthly_returns(level)
    mdd = max_drawdown(level)
    return {
        "Index": name,
        "Start": level.index[0].date(), "End": level.index[-1].date(),
        "Final_Value_of_100": 100.0 * level.iloc[-1] / level.iloc[0],
        "Total_Return": total_return(level), "CAGR": cagr(level), "Annualized_Volatility": ann_vol(level, periods),
        "Sharpe_Ratio": sharpe(level, rf, periods), "Sortino_Ratio": sortino(level, rf, periods),
        "Maximum_Drawdown": mdd["max_drawdown"], "MDD_Peak_Date": mdd["peak_date"].date(), "MDD_Bottom_Date": mdd["bottom_date"].date(),
        "MDD_Recovery_Date": mdd["recovery_date"].date() if mdd["recovery_date"] is not pd.NaT else None,
        "MDD_Days_to_Bottom": mdd["days_to_bottom"], "MDD_Days_to_Recovery": mdd["days_to_recovery"],
        "Calmar_Ratio": calmar(level),
        "Best_Year": f"{ann.idxmax()} ({ann.max():+.1%})", "Worst_Year": f"{ann.idxmin()} ({ann.min():+.1%})",
        "Positive_Year_Ratio": float((ann > 0).mean()), "Monthly_Win_Rate": float((mon > 0).mean()),
    }


def relative_summary(strategy: pd.Series, benchmark: pd.Series, turnover: Optional[pd.DataFrame] = None,
                     periods: int = TRADING_DAYS) -> dict:
    beta, alpha = beta_alpha(strategy, benchmark, periods)
    out = {
        "Annualized_Excess_Return": cagr(strategy) - cagr(benchmark),
        "Tracking_Error": tracking_error(strategy, benchmark, periods),
        "Information_Ratio": information_ratio(strategy, benchmark, periods),
        "Beta_vs_CSI300": beta, "Alpha_vs_CSI300": alpha,
    }
    if turnover is not None and len(turnover) > 1:
        t = turnover.copy()
        t["year"] = pd.to_datetime(t["effective_date"]).dt.year
        yearly = t.iloc[1:].groupby("year")["one_way_turnover"].sum()   # skip initial construction
        n_years = years_between(t["effective_date"].iloc[0], strategy.index[-1])
        out["Average_Annual_Turnover"] = float(t["one_way_turnover"].iloc[1:].sum() / n_years) if n_years > 0 else np.nan
        out["Average_Rebalance_Turnover"] = float(t["one_way_turnover"].iloc[1:].mean())
        out["Maximum_Rebalance_Turnover"] = float(t["one_way_turnover"].iloc[1:].max())
        out["Yearly_Turnover"] = yearly
    return out


def cycle_table(strategy: pd.Series, benchmark: pd.Series, cycles: list[dict], periods: int = TRADING_DAYS) -> pd.DataFrame:
    rows = []
    for c in cycles:
        s0 = pd.Timestamp(c["start"])
        e0 = pd.Timestamp(c["end"]) if c.get("end") else strategy.index[-1]
        s = strategy.loc[s0:e0]
        b = benchmark.loc[s0:e0]
        if len(s) < 20:
            continue
        rows.append({
            "Cycle": c["name"], "Start": s.index[0].date(), "End": s.index[-1].date(),
            "CSI300_Return": total_return(b), "Strategy_Return": total_return(s), "Excess_Return": total_return(s) - total_return(b),
            "CSI300_CAGR": cagr(b), "Strategy_CAGR": cagr(s),
            "CSI300_Volatility": ann_vol(b, periods), "Strategy_Volatility": ann_vol(s, periods),
            "CSI300_Sharpe": sharpe(b, 0.0, periods), "Strategy_Sharpe": sharpe(s, 0.0, periods),
            "CSI300_MaxDD": max_drawdown(b)["max_drawdown"], "Strategy_MaxDD": max_drawdown(s)["max_drawdown"],
        })
    return pd.DataFrame(rows)


def momentum_crashes(strategy: pd.Series, benchmark: pd.Series, window_days: int = 63, top: int = 5) -> pd.DataFrame:
    """Largest trailing-3-month underperformance episodes of the strategy vs the benchmark."""
    rs = strategy / strategy.shift(window_days) - 1.0
    rb = benchmark / benchmark.shift(window_days) - 1.0
    rel = (rs - rb).dropna()
    rows = []
    used = pd.Series(False, index=rel.index)
    for d in rel.sort_values().index:
        if used.loc[d]:
            continue
        lo = rel.index[max(0, rel.index.get_loc(d) - window_days)]
        hi = rel.index[min(len(rel) - 1, rel.index.get_loc(d) + window_days)]
        used.loc[lo:hi] = True
        rows.append({"End_Date": d.date(), "Start_Date": strategy.index[strategy.index.get_loc(d) - window_days].date(),
                     "Strategy_3M_Return": float(rs.loc[d]), "CSI300_3M_Return": float(rb.loc[d]), "Relative_3M": float(rel.loc[d])})
        if len(rows) >= top:
            break
    return pd.DataFrame(rows)
