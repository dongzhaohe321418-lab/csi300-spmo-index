"""S&P Financial Viability screen applied point-in-time to A-share income statements.

S&P U.S. Indices Methodology (Financial Viability): the sum of the most recent four
consecutive quarters' GAAP earnings (net income excluding discontinued operations)
must be positive, as must the most recent quarter.

Mapping to Chinese Accounting Standards (Eastmoney F10 利润表 fields):
    CONTINUED_NETPROFIT  持续经营净利润   (preferred; disclosed since the 2018 CAS format)
    NETPROFIT            净利润           (fallback; total net profit incl. minorities)
Quarterly reports in China are cumulative year-to-date, so single-quarter figures are
differences of consecutive YTD figures within a fiscal year (Q1 = YTD Q1).

Point-in-time rule: a report is usable on reference date R only if its first public
announcement date (NOTICE_DATE) <= R.  If the announcement date is unknown, the
statutory deadline is assumed (Q1: Apr-30, Q2: Aug-31, Q3: Oct-31, Q4: Apr-30 of the
following year) — conservative, never earlier than the true publication date.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

QUARTER_END_MONTHS = (3, 6, 9, 12)
MAX_NOTICE_LAG_DAYS = 200      # annual reports are due within 4 months of the period end


def statutory_deadline(period_end: pd.Timestamp) -> pd.Timestamp:
    m = period_end.month
    y = period_end.year
    if m == 3:
        return pd.Timestamp(year=y, month=4, day=30)
    if m == 6:
        return pd.Timestamp(year=y, month=8, day=31)
    if m == 9:
        return pd.Timestamp(year=y, month=10, day=31)
    return pd.Timestamp(year=y + 1, month=4, day=30)


def quarterly_table(income: pd.DataFrame, field_priority=("CONTINUED_NETPROFIT", "NETPROFIT")) -> pd.DataFrame:
    """Turn cumulative statements into a single-quarter series with availability dates.

    Returns columns: code, period_end, ytd, quarter_ni, field, notice_date, available_date
    """
    if income is None or income.empty:
        return pd.DataFrame(columns=["code", "period_end", "ytd", "quarter_ni", "field", "notice_date", "available_date"])
    df = income.copy()
    df["period_end"] = pd.to_datetime(df["REPORT_DATE"]).dt.normalize()
    df = df[df["period_end"].dt.month.isin(QUARTER_END_MONTHS)]
    df = df[df["period_end"].dt.is_month_end | (df["period_end"].dt.day >= 28)]
    # value & field used
    ytd = pd.Series(np.nan, index=df.index, dtype=float)
    field = pd.Series("", index=df.index, dtype=object)
    for f in field_priority:
        if f in df.columns:
            take = ytd.isna() & df[f].notna()
            ytd[take] = df.loc[take, f].astype(float)
            field[take] = f
    df["ytd"] = ytd
    df["field"] = field
    df["notice_date"] = pd.to_datetime(df["NOTICE_DATE"], errors="coerce") if "NOTICE_DATE" in df.columns else pd.NaT
    df = df.dropna(subset=["ytd"]).sort_values("period_end").drop_duplicates("period_end", keep="last")
    # Eastmoney's NOTICE_DATE for periods before ~2010 is the date of the *following year's*
    # report in which the period appears as a comparative (lag ≈ 13 months). A first announcement
    # can never be later than the statutory deadline (+ a small grace period), so such dates are
    # treated as unknown and replaced by the statutory deadline (conservative: never earlier than
    # the true publication date).
    lag = (df["notice_date"] - df["period_end"]).dt.days
    implausible = df["notice_date"].isna() | (lag > MAX_NOTICE_LAG_DAYS) | (lag < 0)
    df["notice_date_raw"] = df["notice_date"]
    df.loc[implausible, "notice_date"] = pd.NaT
    df["available_date"] = df["notice_date"].fillna(df["period_end"].map(statutory_deadline))
    # single quarter = YTD_t - YTD_{previous quarter of same fiscal year}
    prev_end = df["period_end"] - pd.offsets.QuarterEnd(1)
    prev = df.set_index("period_end")
    prev_ytd = prev_end.map(prev["ytd"]).astype(float)
    prev_avail = prev_end.map(prev["available_date"])
    q1 = df["period_end"].dt.month == 3
    quarter_ni = np.where(q1, df["ytd"], df["ytd"] - prev_ytd.values)
    df["quarter_ni"] = quarter_ni
    df.loc[~q1 & prev_ytd.isna().values, "quarter_ni"] = np.nan
    df["available_date"] = np.where(q1 | prev_avail.isna().values, df["available_date"],
                                    np.maximum(df["available_date"].values, prev_avail.fillna(pd.Timestamp("1900-01-01")).values))
    df["available_date"] = pd.to_datetime(df["available_date"])
    code = str(income["code"].iloc[0]) if "code" in income.columns else ""
    out = df[["period_end", "ytd", "quarter_ni", "field", "notice_date", "available_date"]].copy()
    out.insert(0, "code", code)
    return out.reset_index(drop=True)


@dataclass
class ViabilityResult:
    eligible: bool
    latest_period: Optional[pd.Timestamp]
    latest_available: Optional[pd.Timestamp]
    quarter_ni: float
    four_quarter_ni: float
    n_quarters: int
    field: str
    reason: str


def financial_viability(qtab: pd.DataFrame, ref_date, require_latest_positive: bool = True,
                        require_4q_positive: bool = True) -> ViabilityResult:
    """Evaluate the screen using only reports public on/before ref_date."""
    ref = pd.Timestamp(ref_date).normalize()
    if qtab is None or qtab.empty:
        return ViabilityResult(False, None, None, np.nan, np.nan, 0, "", "no financial statements")
    avail = qtab[(qtab["available_date"] <= ref) & qtab["quarter_ni"].notna()].sort_values("period_end")
    if avail.empty:
        return ViabilityResult(False, None, None, np.nan, np.nan, 0, "", "no report public by reference date")
    latest = avail.iloc[-1]
    # four consecutive quarters ending at the latest available quarter
    needed = [latest["period_end"] - pd.offsets.QuarterEnd(k) for k in range(4)]
    have = avail.set_index("period_end")["quarter_ni"]
    vals = [have.get(pe, np.nan) for pe in needed]
    n_q = int(np.sum(~np.isnan(vals)))
    q_latest = float(vals[0])
    if n_q < 4:
        return ViabilityResult(False, latest["period_end"], latest["available_date"], q_latest, np.nan, n_q,
                               latest["field"], f"only {n_q} consecutive quarters public")
    s4 = float(np.nansum(vals))
    ok = True
    reasons = []
    if require_latest_positive and not (q_latest > 0):
        ok = False
        reasons.append("latest quarter <= 0")
    if require_4q_positive and not (s4 > 0):
        ok = False
        reasons.append("trailing 4Q sum <= 0")
    return ViabilityResult(ok, latest["period_end"], latest["available_date"], q_latest, s4, 4, latest["field"],
                           "pass" if ok else "; ".join(reasons))


def viability_table(qtabs: dict[str, pd.DataFrame], codes, ref_date, **kw) -> pd.DataFrame:
    rows = []
    for c in codes:
        r = financial_viability(qtabs.get(c), ref_date, **kw)
        rows.append((c, r.eligible, r.latest_period, r.latest_available, r.quarter_ni, r.four_quarter_ni, r.n_quarters, r.field, r.reason))
    return pd.DataFrame(rows, columns=["code", "financial_eligible", "fin_latest_period", "fin_latest_available",
                                       "quarter_net_income", "four_quarter_net_income", "fin_n_quarters", "fin_field", "fin_reason"])
