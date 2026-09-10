"""Daily index calculation from the rebalance holdings (divisor / index-shares method).

At the close of each implementation date E_k the strategy holds index shares

    n_i = L(E_k) × w_i / P_i(E_k)

where P_i is the corporate-action adjusted price (price-return basis for the price index,
total-return basis for the TR index).  Between rebalances the shares are fixed, so

    Level_t(close) = Σ n_i × P_i,t(close)
    Level_t(open)  = Σ n_i × P_i,t(open),   High/Low analogously.

Open/High/Low are therefore built from the constituents' own O/H/L with the same index
shares (the standard portfolio-OHLC approximation: because constituent highs and lows
are not simultaneous, the aggregated High slightly overstates and the aggregated Low
slightly understates the true intraday index extremes; Open and Close are exact).
Suspended securities carry their last close in all four fields.

Parent-membership rule: when a holding is removed from the CSI 300 (effective date D) it
leaves the strategy at the close of D-1 and its value is redistributed pro rata to the
remaining holdings.  The engine asserts every day that all holdings are CSI 300 members.

Net index: at each implementation date the one-way turnover is charged with the
historical A-share cost schedule (commission, stamp duty, transfer fee, slippage).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .utils import TradingCalendar

log = logging.getLogger(__name__)


@dataclass
class Rebalance:
    reference_date: pd.Timestamp
    effective_date: pd.Timestamp
    weights: pd.Series               # code -> final weight (sums to 1)
    meta: dict = field(default_factory=dict)


class MembershipError(RuntimeError):
    pass


def _rate(schedule: list[dict], d: pd.Timestamp, key: str = "rate") -> float:
    rate = 0.0
    for row in sorted(schedule, key=lambda r: r["from"]):
        if pd.Timestamp(row["from"]) <= d:
            rate = float(row.get(key, 0.0))
    return rate


def cost_rates(cfg_costs: dict, d: pd.Timestamp) -> tuple[float, float]:
    """(buy_rate, sell_rate) as fractions of traded value on date d."""
    comm = _rate(cfg_costs["commission_schedule"], d)
    transfer = _rate(cfg_costs.get("transfer_fee_schedule", []), d)
    slip = float(cfg_costs.get("slippage_per_side", 0.0))
    duty_buy = _rate(cfg_costs["stamp_duty_schedule"], d, "buy")
    duty_sell = _rate(cfg_costs["stamp_duty_schedule"], d, "sell")
    return comm + transfer + slip + duty_buy, comm + transfer + slip + duty_sell


class IndexEngine:
    def __init__(self, panel: dict[str, pd.DataFrame], cal: TradingCalendar, membership: pd.DataFrame, cfg: dict):
        self.cal = cal
        self.cfg = cfg
        self.dates = panel["adj_close_pr"].index
        self.codes = list(panel["adj_close_pr"].columns)
        self.col = {c: i for i, c in enumerate(self.codes)}
        self.P = {
            "pr": {k: panel[f"adj_{k}_pr"].values for k in ("close", "open", "high", "low")},
            "tr": {"close": panel["adj_close_tr"].values},
        }
        self.traded = panel["traded"].values.astype(bool)
        # membership: for each code, list of (start, end) intervals
        mem = membership.copy()
        mem["start_date"] = pd.to_datetime(mem["start_date"])
        mem["end_date"] = pd.to_datetime(mem["end_date"])
        self.membership = mem
        # removal map: effective date -> set(codes removed that day)
        rem = mem.dropna(subset=["end_date"])
        self.removals: dict[pd.Timestamp, set] = {}
        for d, g in rem.groupby("end_date"):
            self.removals[pd.Timestamp(d)] = set(g["code"])

    def members_at(self, d: pd.Timestamp) -> set:
        m = self.membership
        mask = (m["start_date"] <= d) & (m["end_date"].isna() | (m["end_date"] > d))
        return set(m.loc[mask, "code"])

    def run(self, rebalances: list[Rebalance], basis: str = "pr", base_level: float = 1000.0) -> dict:
        Pc = self.P[basis]["close"]
        has_ohlc = basis == "pr"
        Po = self.P["pr"]["open"] if has_ohlc else Pc
        Ph = self.P["pr"]["high"] if has_ohlc else Pc
        Pl = self.P["pr"]["low"] if has_ohlc else Pc
        rebs = sorted(rebalances, key=lambda r: r.effective_date)
        reb_by_date = {r.effective_date: r for r in rebs}
        t0 = self.dates.get_loc(rebs[0].effective_date)
        costs_cfg = self.cfg["costs"]

        levels = []          # (date, open, high, low, close, net_close, n_holdings)
        turnover_rows = []
        holdings_daily = {}  # effective date -> weights actually implemented
        shares: dict[str, float] = {}
        gross = base_level
        net = base_level

        def _set_shares(level: float, weights: pd.Series, ti: int) -> dict:
            out = {}
            for c, w in weights.items():
                j = self.col[c]
                p = Pc[ti, j]
                if not np.isfinite(p) or p <= 0:
                    raise ValueError(f"no price for {c} on {self.dates[ti].date()}")
                out[c] = level * w / p
            return out

        # initial portfolio at close of first effective date
        r0 = rebs[0]
        self._assert_members(r0.weights.index, r0.effective_date)
        shares = _set_shares(gross, r0.weights, t0)
        turnover_rows.append({"reference_date": r0.reference_date, "effective_date": r0.effective_date,
                              "n_holdings": len(shares), "one_way_turnover": 1.0, "buy_turnover": 1.0, "sell_turnover": 0.0,
                              "cost_rate": cost_rates(costs_cfg, r0.effective_date)[0], "cost": cost_rates(costs_cfg, r0.effective_date)[0]})
        net = gross * (1.0 - turnover_rows[-1]["cost"])
        levels.append((self.dates[t0], gross, gross, gross, gross, net, len(shares)))
        holdings_daily[r0.effective_date] = r0.weights.copy()

        for ti in range(t0 + 1, len(self.dates)):
            d = self.dates[ti]
            # 1) parent-index removals effective today -> drop at yesterday's close, redistribute
            removed_today = self.removals.get(d, set()) & set(shares)
            if removed_today:
                vals = {c: shares[c] * Pc[ti - 1, self.col[c]] for c in shares}
                total = sum(vals.values())
                keep_total = total - sum(vals[c] for c in removed_today)
                if keep_total <= 0:
                    raise ValueError(f"all holdings removed on {d.date()}")
                factor = total / keep_total
                shares = {c: s * factor for c, s in shares.items() if c not in removed_today}
                log.info("%s: %s left the CSI 300 -> dropped from strategy, weight redistributed", d.date(), sorted(removed_today))
            idx = np.array([self.col[c] for c in shares])
            n = np.array([shares[c] for c in shares])
            pc = Pc[ti, idx]
            if np.isnan(pc).any():
                bad = [c for c, v in zip(shares, pc) if np.isnan(v)]
                raise ValueError(f"missing price for {bad} on {d.date()} (delisted while in the parent index?)")
            tr = self.traded[ti, idx]
            po = np.where(tr, Po[ti, idx], pc)
            ph = np.where(tr, Ph[ti, idx], pc)
            pl = np.where(tr, Pl[ti, idx], pc)
            prev_gross = gross
            gross = float((n * pc).sum())
            o, h, l = float((n * po).sum()), float((n * ph).sum()), float((n * pl).sum())
            net = net * gross / prev_gross
            # 2) rebalance at today's close?
            if d in reb_by_date:
                rb = reb_by_date[d]
                self._assert_members(rb.weights.index, d)
                drift = pd.Series(n * pc / gross, index=list(shares))
                new_w = rb.weights
                allc = drift.index.union(new_w.index)
                dw = new_w.reindex(allc).fillna(0.0) - drift.reindex(allc).fillna(0.0)
                buy = float(dw.clip(lower=0).sum())
                sell = float((-dw).clip(lower=0).sum())
                br, sr = cost_rates(costs_cfg, d)
                cost = buy * br + sell * sr
                net = net * (1.0 - cost)
                shares = _set_shares(gross, new_w, ti)
                turnover_rows.append({"reference_date": rb.reference_date, "effective_date": d, "n_holdings": len(shares),
                                      "one_way_turnover": 0.5 * (buy + sell), "buy_turnover": buy, "sell_turnover": sell,
                                      "cost_rate": br + sr, "cost": cost})
                holdings_daily[d] = new_w.copy()
            levels.append((d, o, h, l, gross, net, len(shares)))

        lev = pd.DataFrame(levels, columns=["Date", "Open", "High", "Low", "Close", "Net_Close", "n_holdings"]).set_index("Date")
        to = pd.DataFrame(turnover_rows)
        return {"levels": lev, "turnover": to, "holdings": holdings_daily}

    def _assert_members(self, codes, d: pd.Timestamp) -> None:
        members = self.members_at(d)
        bad = [c for c in codes if c not in members]
        if bad:
            raise MembershipError(f"{len(bad)} holdings with weight > 0 are not CSI 300 members on {pd.Timestamp(d).date()}: {bad[:10]}")


def benchmark_levels(index_daily: pd.DataFrame, start: pd.Timestamp, base_level: float = 1000.0) -> pd.DataFrame:
    """Rebase the official CSI 300 OHLC to base_level at the strategy start date."""
    df = index_daily.set_index("date").sort_index()
    df = df[df.index >= start]
    scale = base_level / df["close"].iloc[0]
    out = pd.DataFrame({
        "Open": df["open"] * scale, "High": df["high"] * scale, "Low": df["low"] * scale, "Close": df["close"] * scale,
    })
    out.index.name = "Date"
    return out
