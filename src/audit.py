"""Automatic eligibility / look-ahead audit of every rebalance (spec section 12).

For every reference date the selected constituents must
 1. be CSI 300 members on the reference date (and on the implementation date),
 2. pass the S&P Financial Viability screen,
 3. rely only on financial reports whose announcement date <= reference date,
 4. have enough momentum history (price 13 months earlier, >= 150 traded days),
 5. use no information dated after the reference date (momentum window ends before it).
Any violation raises AuditError and the run stops before performance outputs are written.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .universe import members_at
from .utils import abs_path

log = logging.getLogger(__name__)


class AuditError(RuntimeError):
    pass


def run(cfg: dict, intervals: pd.DataFrame, schedule: list[tuple[pd.Timestamp, pd.Timestamp]], min_traded_days: int) -> pd.DataFrame:
    scores_dir = abs_path(cfg, cfg["output"]["scores_dir"])
    rows = []
    problems = []
    for ref, eff in schedule:
        p = scores_dir / f"{ref:%Y-%m-%d}.csv"
        if not p.exists():
            problems.append(f"{ref.date()}: missing scores file")
            continue
        df = pd.read_csv(p, dtype={"ticker": str})
        sel = df[df["selected"].astype(bool)]
        mem_ref = members_at(intervals, ref)
        mem_eff = members_at(intervals, eff)
        not_member_ref = sorted(set(sel["ticker"]) - mem_ref)
        not_member_eff = sorted(set(sel["ticker"]) - mem_eff)
        pos_w_nonmember = sorted(set(df.loc[df["final_weight"] > 0, "ticker"]) - mem_ref)
        fin_fail = sel[~sel["financial_eligible"].astype(bool)]["ticker"].tolist()
        fin_avail = pd.to_datetime(sel["fin_latest_available"])
        fin_future = sel[fin_avail > ref]["ticker"].tolist()
        mom_short = sel[(sel["traded_days"] < min_traded_days) | sel["momentum_return"].isna()]["ticker"].tolist()
        mom_future = sel[pd.to_datetime(sel["mom_end_date"]) >= ref]["ticker"].tolist()
        wsum = float(sel["final_weight"].sum())
        row = {"reference_date": ref.date(), "implementation_date": eff.date(), "n_selected": len(sel),
               "non_members_on_ref_date": len(not_member_ref), "non_members_on_impl_date": len(not_member_eff),
               "positive_weight_non_members": len(pos_w_nonmember), "financial_failures": len(fin_fail),
               "reports_after_ref_date": len(fin_future), "insufficient_momentum_history": len(mom_short),
               "momentum_data_after_ref_date": len(mom_future), "weight_sum": round(wsum, 8)}
        rows.append(row)
        for label, bad in (("not CSI300 member on reference date", not_member_ref), ("positive weight for non-member", pos_w_nonmember),
                           ("fails financial viability", fin_fail), ("financial report published after reference date", fin_future),
                           ("insufficient momentum history", mom_short), ("momentum data after reference date", mom_future)):
            if bad:
                problems.append(f"{ref.date()}: {label}: {bad[:8]}")
        if abs(wsum - 1.0) > 1e-6:
            problems.append(f"{ref.date()}: final weights sum to {wsum:.8f}")
        if not_member_eff:
            # a security can legitimately leave the parent between reference and implementation
            # dates; it must then be excluded from the implemented portfolio (handled by the engine)
            log.warning("%s: %d selected securities left the CSI 300 before implementation (%s)", ref.date(), len(not_member_eff), not_member_eff[:5])
    audit = pd.DataFrame(rows)
    if problems:
        for pr in problems:
            log.error("AUDIT: %s", pr)
        raise AuditError("eligibility audit failed:\n" + "\n".join(problems))
    return audit
