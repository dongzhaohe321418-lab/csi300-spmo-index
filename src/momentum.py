"""S&P Momentum Indices — risk-adjusted 12-1 month price momentum.

For reference date R (last trading day of Feb / Aug):

    end   = last trading day on/before (R - 1 calendar month)
    start = last trading day on/before (R - 13 calendar months)
    Momentum Return  = P_adj(end) / P_adj(start) - 1          (split/rights adjusted, ex-dividend)
    Volatility       = StdDev( daily price returns on traded days in (start, end] )
    Risk-Adjusted Momentum (RAM) = Momentum Return / Volatility

Eligibility for a momentum value: the security must have traded on at least
`min_traded_days` (150) days inside the window and must have a price on/before `start`.
Only information available on R is used (all prices are dated <= end < R).
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .utils import TradingCalendar, months_before

log = logging.getLogger(__name__)


def momentum_window(ref_date, cal: TradingCalendar, lookback_months: int = 12, skip_months: int = 1) -> tuple[pd.Timestamp, pd.Timestamp]:
    ref = pd.Timestamp(ref_date)
    end = cal.last_on_or_before(months_before(ref, skip_months))
    start = cal.last_on_or_before(months_before(ref, skip_months + lookback_months))
    return start, end


def compute_momentum(adj_close: pd.DataFrame, ret: pd.DataFrame, traded: pd.DataFrame, ref_date, cal: TradingCalendar,
                     codes: Optional[list[str]] = None, lookback_months: int = 12, skip_months: int = 1,
                     min_traded_days: int = 150) -> pd.DataFrame:
    """Vectorised momentum for all `codes` (columns) at one reference date.

    adj_close / ret / traded: wide date x code panels aligned to the trading calendar
    (prices forward-filled on suspension days, returns 0 there, traded 0/1).
    """
    start, end = momentum_window(ref_date, cal, lookback_months, skip_months)
    cols = list(codes) if codes is not None else list(adj_close.columns)
    cols = [c for c in cols if c in adj_close.columns]
    p_start = adj_close.loc[start, cols]
    p_end = adj_close.loc[end, cols]
    win = slice(cal.next(start, 1), end)              # (start, end]
    r = ret.loc[win, cols]
    t = traded.loc[win, cols].astype(bool)
    r_traded = r.where(t)                            # only traded days enter the volatility
    n_traded = t.sum(axis=0)
    vol = r_traded.std(axis=0, ddof=1)
    mom = p_end / p_start - 1.0
    has_start = p_start.notna()
    ok = has_start & p_end.notna() & (n_traded >= min_traded_days) & vol.gt(0)
    ram = (mom / vol).where(ok)
    out = pd.DataFrame({
        "code": cols,
        "mom_start_date": start,
        "mom_end_date": end,
        "price_start": p_start.values,
        "price_end": p_end.values,
        "momentum_return": mom.where(has_start & p_end.notna()).values,
        "volatility": vol.where(ok).values,
        "traded_days": n_traded.values.astype(int),
        "RAM": ram.values,
        "momentum_eligible": ok.values,
    })
    out["mom_reason"] = np.where(~has_start.values, "no price 13 months before reference date",
                                 np.where(n_traded.values < min_traded_days, f"traded < {min_traded_days} days in window",
                                          np.where(~vol.gt(0).values, "zero volatility", "ok")))
    return out
