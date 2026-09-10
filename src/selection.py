"""Momentum score and constituent selection (S&P Momentum Indices methodology).

Z-score   : cross-sectional over the eligible universe (CSI300 member AND financially
            eligible AND momentum-eligible), winsorised at ±3.
Score     : 1 + Z  if Z > 0  else  1 / (1 - Z)
Target N  : round(top_fraction × eligible count)  (e.g. 280 eligible -> 56)
Buffer    : securities ranked within the top 80% of N are selected automatically; current
            constituents ranked within the top 120% of N are then retained in rank order
            until N is reached; remaining slots are filled by rank.  A current
            constituent that is no longer a CSI300 member or fails the screens cannot be
            retained (parent membership has priority over the buffer).
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def zscore(x: pd.Series, winsor: float = 3.0) -> pd.Series:
    mu = x.mean()
    sd = x.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=x.index)
    z = (x - mu) / sd
    return z.clip(lower=-winsor, upper=winsor)


def momentum_score(z: pd.Series) -> pd.Series:
    return pd.Series(np.where(z > 0, 1.0 + z, 1.0 / (1.0 - z)), index=z.index)


def target_count(n_eligible: int, top_fraction: float = 0.20, rounding: str = "nearest") -> int:
    x = n_eligible * top_fraction
    if rounding == "up":
        return int(math.ceil(x))
    if rounding == "down":
        return int(math.floor(x))
    return int(math.floor(x + 0.5))


def select_constituents(scores: pd.DataFrame, current: Iterable[str], top_fraction: float = 0.20,
                        auto_frac: float = 0.80, retain_frac: float = 1.20, rounding: str = "nearest") -> pd.DataFrame:
    """`scores` must contain code, RAM, eligible (bool). Adds z_score, momentum_score, rank, selected."""
    df = scores.copy()
    elig = df["eligible"].fillna(False).astype(bool)
    z = pd.Series(np.nan, index=df.index)
    z[elig] = zscore(df.loc[elig, "RAM"].astype(float))
    df["z_score"] = z
    df["momentum_score"] = np.nan
    df.loc[elig, "momentum_score"] = momentum_score(z[elig]).values
    # rank: highest score first; ties broken by RAM then code for determinism
    ranked = df[elig].sort_values(["momentum_score", "RAM", "code"], ascending=[False, False, True])
    df["rank"] = np.nan
    df.loc[ranked.index, "rank"] = np.arange(1, len(ranked) + 1)
    n_elig = int(elig.sum())
    n_target = target_count(n_elig, top_fraction, rounding)
    auto_cut = math.floor(auto_frac * n_target)
    retain_cut = math.floor(retain_frac * n_target)
    cur = set(current)
    selected: list = []
    for idx in ranked.index:
        if df.at[idx, "rank"] <= auto_cut:
            selected.append(idx)
    for idx in ranked.index:                       # retain current members inside the buffer
        if len(selected) >= n_target:
            break
        if idx in selected:
            continue
        if df.at[idx, "code"] in cur and df.at[idx, "rank"] <= retain_cut:
            selected.append(idx)
    for idx in ranked.index:                       # fill by rank
        if len(selected) >= n_target:
            break
        if idx not in selected:
            selected.append(idx)
    df["selected"] = False
    df.loc[selected, "selected"] = True
    df["buffer_retained"] = False
    df.loc[[i for i in selected if df.at[i, "rank"] > auto_cut and df.at[i, "code"] in cur], "buffer_retained"] = True
    df.attrs.update({"n_eligible": n_elig, "n_target": n_target, "auto_cut": auto_cut, "retain_cut": retain_cut})
    return df
