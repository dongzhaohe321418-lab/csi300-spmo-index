"""Corporate-action adjusted daily prices and the wide price panel.

For every security we build two adjusted close series from the raw Sina bars:

* ``adj_close_pr`` — price-return series: adjusted for bonus/transfer shares (送转) and
  rights issues (配股) but **not** for cash dividends.  Momentum ("12-1 month price
  change") and the strategy *price* index are computed on it, which makes the strategy
  comparable with the CSI 300 price index (000300).
* ``adj_close_tr`` — total-return series: cash dividends reinvested on the ex-date at the
  exchange reference price (the usual 后复权 convention); used for the total-return
  variant compared with CSI 300 Total Return (H00300).

Daily returns are anchored on the exchange's ex-date reference price (`prevclose`, only
populated by Sina on ex-rights/ex-dividend days):

    ref_tr = (C_{t-1} - D + r·P_r) / (1 + b + r)           (exchange reference price)
    ref_pr = ref_tr + D / (1 + b + r)                       (dividend added back)
    ret_tr = C_t / ref_tr - 1,   ret_pr = C_t / ref_pr - 1

where D = cash dividend per share (pre-tax), b = bonus+transfer shares per share and
(r, P_r) = rights-issue ratio and price.  D, b come from the Eastmoney dividend table,
(r, P_r) from the market-wide rights-issue table.  Suspension days simply carry the last
price (zero return, not a traded day).

Float-adjusted market cap proxy: close × 已上市流通A股 (listed A-shares from the share
structure history, as of the date).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .data import DataHub, NoData

log = logging.getLogger(__name__)

PANEL_FIELDS = ("adj_close_pr", "adj_open_pr", "adj_high_pr", "adj_low_pr", "adj_close_tr", "ret_pr", "ret_tr",
                "traded", "fmc", "close")

# Holders that CSIndex treats as free-float even above 5% (institutional investors).
INSTITUTIONAL_RE = re.compile(
    r"基金|保险|社保|养老|年金|QFII|证券金融|汇金资产管理|资产管理计划|资管计划|集合(资产|计划)|信托|理财|投资组合|"
    r"香港中央结算有限公司|HKSCC|中央结算|ETF|LOF|证券投资|Fund|FUND|Investment|Investors|Asset|Capital|Securities|"
    r"Nominees|NOMINEES|Bank|BANK|Insurance|Life|Pension|人寿|财险|健康险|再保险|银行股份有限公司-|证券股份有限公司-"
)
CSINDEX_TIERS = [(0.15, None), (0.20, 0.20), (0.30, 0.30), (0.40, 0.40), (0.50, 0.50), (0.60, 0.60), (0.70, 0.70), (0.80, 0.80), (1.01, 1.00)]


def tiered_ratio(ff_ratio: float) -> float:
    """CSIndex 分级靠档: ratio <= 15% -> rounded up to the next 1%; otherwise 20/30/.../80/100% tiers."""
    if not np.isfinite(ff_ratio):
        return np.nan
    r = min(max(ff_ratio, 0.0), 1.0)
    if r <= 0.15:
        return max(np.ceil(r * 100.0 - 1e-9) / 100.0, 0.01)
    for upper, tier in CSINDEX_TIERS[1:]:
        if r <= upper:
            return tier
    return 1.0


def non_free_shares(holders: pd.DataFrame, total_shares_by_date: Optional[pd.Series] = None, threshold_pct: float = 5.0) -> pd.DataFrame:
    """Per shareholder snapshot: A-shares held by non-free-float holders (>= threshold, not institutional).

    Returns period_end, available_date, non_free_shares (aligned to the announcement date, or
    period_end + 60 days when missing — conservative).
    """
    cols = ["period_end", "available_date", "non_free_shares"]
    if holders is None or holders.empty:
        return pd.DataFrame(columns=cols)
    h = holders.copy()
    # only holdings that are part of the listed A-share float can be subtracted from it:
    # skip H/B shares and the pre-reform non-tradable classes (国有股/法人股/...) as well as
    # pure restricted (限售) positions, which are not in 已上市流通A股 to begin with
    cls = h["share_class"].astype(str)
    is_a_float = cls.str.contains("流通A股|A股", regex=True) & ~cls.str.fullmatch(r"限售[^,，]*")
    h = h[is_a_float | (cls.isin(["nan", "", "None"]))]
    h = h[~h["holder"].str.contains(INSTITUTIONAL_RE, na=False)]
    # Sina leaves pct or shares blank in some snapshots: fill one from the other via total shares
    if total_shares_by_date is not None and len(total_shares_by_date):
        tot = pd.Series([total_shares_by_date.asof(pe) if pe >= total_shares_by_date.index[0] else total_shares_by_date.iloc[0]
                         for pe in h["period_end"]], index=h.index, dtype=float)
        h["pct"] = h["pct"].fillna(h["shares"] / tot * 100.0)
        h["shares"] = h["shares"].fillna(h["pct"] / 100.0 * tot)
    h = h[h["pct"].fillna(0) >= threshold_pct]
    rows = []
    for pe, g in holders.groupby("period_end"):
        gg = h[h["period_end"] == pe]
        ann = g["ann_date"].dropna()
        avail = ann.min() if len(ann) else pe + pd.Timedelta(days=60)
        rows.append((pe, avail, float(gg["shares"].fillna(0).sum())))
    out = pd.DataFrame(rows, columns=cols).sort_values("available_date").reset_index(drop=True)
    return out


def adjust_security(daily: pd.DataFrame, dividends: pd.DataFrame, rights: pd.DataFrame,
                    shares: Optional[pd.DataFrame], cal_days: pd.DatetimeIndex,
                    holders: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Return a per-security frame indexed by its own trading days."""
    df = daily.sort_values("date").drop_duplicates("date").copy()
    df = df[df["date"].isin(cal_days)]
    df = df[df["close"] > 0].reset_index(drop=True)
    if df.empty:
        return df
    prev_close = df["close"].shift(1)
    # corporate actions keyed by ex-date
    div = pd.Series(dtype=float)
    bon = pd.Series(dtype=float)
    if dividends is not None and not dividends.empty:
        d = dividends.dropna(subset=["ex_date"]).copy()
        d = d[d["status"].astype(str).str.contains("实施|分配|方案实施|完成", regex=True) | d["status"].isna() | (d["status"] == "")]
        d["ex_date"] = pd.to_datetime(d["ex_date"]).dt.normalize()
        g = d.groupby("ex_date").agg(cash=("cash_div_per10", "sum"), bonus=("bonus_per10", "sum"))
        div = g["cash"].fillna(0.0) / 10.0
        bon = g["bonus"].fillna(0.0) / 10.0
    rr = pd.Series(dtype=float)
    rp = pd.Series(dtype=float)
    if rights is not None and not rights.empty:
        r = rights.dropna(subset=["record_date"]).copy()
        # ex-rights day = first trading day after the record date
        pos = cal_days.searchsorted(pd.to_datetime(r["record_date"]).values, side="right")
        pos = np.minimum(pos, len(cal_days) - 1)
        r["ex_date"] = cal_days[pos]
        g = r.groupby("ex_date").agg(ratio=("rights_per_share", "sum"), price=("rights_price", "mean"))
        rr, rp = g["ratio"], g["price"]
    D = df["date"].map(div).fillna(0.0).values
    B = df["date"].map(bon).fillna(0.0).values
    R = df["date"].map(rr).fillna(0.0).values
    PR = df["date"].map(rp).fillna(0.0).values
    prevclose = df["prevclose"].values.astype(float)
    close = df["close"].values.astype(float)
    pc = prev_close.values.astype(float)
    has_ref = np.isfinite(prevclose) & (prevclose > 0)
    has_ca = (D > 0) | (B > 0) | (R > 0)
    denom = 1.0 + B + R
    # total-return reference
    ref_tr = np.where(has_ref, prevclose, np.where(has_ca, (pc - D + R * PR) / denom, pc))
    # price-return reference: add the dividend back
    ref_pr = np.where(has_ca, ref_tr + D / denom, ref_tr)
    # exchange reference present but no corporate-action record (e.g. delisted security
    # without dividend data): classify by size — small gaps are dividends, big gaps splits
    only_ref = has_ref & ~has_ca & np.isfinite(pc)
    ratio = np.where(np.isfinite(pc) & (pc > 0), prevclose / pc, 1.0)
    ref_pr = np.where(only_ref & (ratio > 0.94), pc, ref_pr)          # dividend-like gap
    ref_pr = np.where(only_ref & (ratio <= 0.94), ref_tr, ref_pr)     # split-like gap
    ret_tr = close / ref_tr - 1.0
    ret_pr = close / ref_pr - 1.0
    ret_tr[0] = 0.0
    ret_pr[0] = 0.0
    ret_tr = np.where(np.isfinite(ret_tr), ret_tr, 0.0)
    ret_pr = np.where(np.isfinite(ret_pr), ret_pr, 0.0)
    df["ret_tr"] = ret_tr
    df["ret_pr"] = ret_pr
    base = close[0]
    df["adj_close_tr"] = base * np.cumprod(1.0 + ret_tr)
    df["adj_close_pr"] = base * np.cumprod(1.0 + ret_pr)
    for f in ("open", "high", "low"):
        df[f"adj_{f}_pr"] = df["adj_close_pr"] * (df[f].astype(float) / df["close"].astype(float))
    df["traded"] = (df["volume"].fillna(0) > 0).astype(np.int8)
    # shares outstanding (listed A shares) as of date
    if shares is not None and not shares.empty:
        s = shares.dropna(subset=["listed_a_shares"]).sort_values("change_date")
        s = s[s["listed_a_shares"] > 0]
        if not s.empty:
            merged = pd.merge_asof(df[["date"]], s[["change_date", "listed_a_shares"]].rename(columns={"change_date": "date"}),
                                   on="date", direction="backward")
            la = merged["listed_a_shares"].values
            la = np.where(np.isnan(la), s["listed_a_shares"].iloc[0], la)   # before first record: earliest known
            df["listed_a_shares"] = la
        else:
            df["listed_a_shares"] = np.nan
    else:
        df["listed_a_shares"] = np.nan
    # free float: listed A shares minus non-free holders (state / founders / strategic >= 5%),
    # ratio rounded with CSIndex's tiers; falls back to listed A shares when no holder data
    df["non_free_shares"] = 0.0
    if holders is not None and not holders.empty:
        tot = None
        if shares is not None and not shares.empty:
            tot = shares.dropna(subset=["total_shares"]).set_index("change_date")["total_shares"].sort_index()
        nf = non_free_shares(holders, tot)
        if not nf.empty:
            m = pd.merge_asof(df[["date"]], nf[["available_date", "non_free_shares"]].rename(columns={"available_date": "date"}),
                              on="date", direction="backward")
            v = m["non_free_shares"].values
            v = np.where(np.isnan(v), nf["non_free_shares"].iloc[0], v)   # before first snapshot: earliest known
            df["non_free_shares"] = v
    ff = (df["listed_a_shares"] - df["non_free_shares"]).clip(lower=0)
    ratio = (ff / df["listed_a_shares"]).where(df["listed_a_shares"] > 0)
    df["ff_ratio"] = ratio
    df["ff_ratio_tiered"] = ratio.map(tiered_ratio)
    df["free_float_shares"] = df["listed_a_shares"] * df["ff_ratio_tiered"]
    df["fmc"] = df["close"] * df["free_float_shares"]
    df.loc[df["fmc"].isna() & df["listed_a_shares"].notna(), "fmc"] = df["close"] * df["listed_a_shares"]
    return df


def build_all(hub: DataHub, codes: Iterable[str], cal_days: pd.DatetimeIndex, out_dir: Path) -> pd.DataFrame:
    """Adjust every security and write data/processed/prices/{code}.parquet. Returns a status table."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rights_all = hub.rights_issues()
    codes = list(codes)
    # Free-float refinement from top-10 shareholder data is only applied when it is available
    # for (almost) every security, so that all constituents are treated consistently.
    holder_dir = hub.raw.path("sina", "holders", "x").parent
    n_holder = len(list(holder_dir.glob("*.parquet"))) if holder_dir.exists() else 0
    use_holders = n_holder >= 0.9 * len(codes)
    log.info("float market cap method: %s (shareholder files for %d/%d securities)",
             "listed A-shares minus non-free holders (CSIndex tiers)" if use_holders else "listed A-shares (流通A股)", n_holder, len(codes))
    (out_dir.parent / "fmc_method.txt").write_text(
        ("free_float_holders" if use_holders else "listed_a_shares") + f"\t{n_holder}/{len(codes)} securities with shareholder data\n", encoding="utf-8")
    rows = []
    for i, c in enumerate(codes, 1):
        try:
            daily = hub.stock_daily(c)
        except NoData:
            rows.append((c, "no_prices", 0, None, None))
            continue
        try:
            div = hub.dividends(c)
        except NoData:
            div = None
        try:
            sh = hub.shares(c)
        except NoData:
            sh = None
        hold = None
        if use_holders:
            try:
                hold = hub.holders(c)
            except Exception as exc:  # noqa: BLE001 - holder data is an enhancement, never fatal
                log.warning("holders unavailable for %s: %s", c, exc)
        adj = adjust_security(daily, div, rights_all[rights_all["code"] == c], sh, cal_days, holders=hold)
        if adj.empty:
            rows.append((c, "empty", 0, None, None))
            continue
        adj.to_parquet(out_dir / f"{c}.parquet", index=False)
        rows.append((c, "ok", len(adj), adj["date"].min(), adj["date"].max()))
        if i % 100 == 0:
            log.info("adjusted %d/%d securities", i, len(list(codes)))
    return pd.DataFrame(rows, columns=["code", "status", "n_days", "first_date", "last_date"])


def load_panel(out_dir: Path, cal_days: pd.DatetimeIndex, fields=PANEL_FIELDS) -> dict[str, pd.DataFrame]:
    """Wide (date x code) matrices for the requested fields, aligned to the trading calendar.
    Price fields are forward-filled over suspension days (traded stays 0); returns are 0 there."""
    frames = {f: {} for f in fields}
    for p in sorted(out_dir.glob("*.parquet")):
        code = p.stem
        df = pd.read_parquet(p).set_index("date")
        for f in fields:
            if f in df.columns:
                frames[f][code] = df[f]
    panel = {}
    for f in fields:
        wide = pd.DataFrame(frames[f]).reindex(cal_days)
        if f in ("ret_pr", "ret_tr"):
            # a suspended day has no return; but keep NaN before listing / after delisting
            listed = wide.notna().cummax() & wide[::-1].notna().cummax()[::-1]
            wide = wide.where(~listed | wide.notna(), 0.0)
        elif f == "traded":
            wide = wide.fillna(0).astype(np.int8)
        else:
            wide = wide.ffill()
            # do not extend beyond delisting: mask dates after the last real observation
            last_obs = pd.DataFrame(frames[f]).reindex(cal_days).notna()[::-1].cummax()[::-1]
            wide = wide.where(last_obs)
        panel[f] = wide.sort_index(axis=1)
    return panel
