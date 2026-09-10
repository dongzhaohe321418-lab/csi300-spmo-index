"""SPMO weighting: float-adjusted market cap × momentum score, with company weight caps.

    raw_i   = FMC_i × MomentumScore_i ;  w_i = raw_i / Σ raw
    cap_i   = min(cap_absolute, cap_multiple × parent_weight_i)

`parent_weight_i` is the security's market-cap weight in the parent index (CSI 300),
computed here with the same FMC proxy for all 300 members on the reference date.
Weights above their cap are set to the cap and the excess is redistributed
proportionally to the uncapped securities; the procedure iterates until no cap is
breached.  If the caps are jointly infeasible (Σ cap_i < 100%), the multiple is raised in
increments of 1 until feasible — this relaxation is recorded in the output.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cap_weights(raw: pd.Series, caps: pd.Series, max_iter: int = 200, tol: float = 1e-12) -> tuple[pd.Series, int]:
    w = raw / raw.sum()
    capped = pd.Series(False, index=w.index)
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        breach = (w > caps + tol) & ~capped
        if not breach.any():
            break
        capped |= breach
        excess = (w[capped] - caps[capped]).sum()
        w[capped] = caps[capped]
        free = ~capped
        if free.sum() == 0 or w[free].sum() <= 0:
            break
        w[free] = w[free] + excess * w[free] / w[free].sum()
    return w, n_iter


def compute_weights(selected: pd.DataFrame, cap_absolute: float = 0.09, cap_multiple: float = 3.0,
                    max_iter: int = 200) -> pd.DataFrame:
    """`selected` needs columns code, fmc, momentum_score, csi300_weight (parent weight)."""
    df = selected.copy()
    raw = (df["fmc"].astype(float) * df["momentum_score"].astype(float))
    if (raw <= 0).any() or raw.isna().any():
        bad = df.loc[(raw <= 0) | raw.isna(), "code"].tolist()
        raise ValueError(f"non-positive raw weight for {bad}")
    df["raw_weight"] = raw / raw.sum()
    mult = float(cap_multiple)
    caps = np.minimum(cap_absolute, mult * df["csi300_weight"].astype(float))
    relaxed = 0
    while caps.sum() < 1.0 - 1e-9 and relaxed < 1000:
        mult += 1.0
        relaxed += 1
        caps = np.minimum(cap_absolute, mult * df["csi300_weight"].astype(float))
    if caps.sum() < 1.0 - 1e-9:  # only possible if cap_absolute * n < 1
        caps = pd.Series(1.0, index=df.index)
    w, n_iter = cap_weights(df["raw_weight"], pd.Series(caps.values, index=df.index), max_iter=max_iter)
    df["weight_cap"] = caps.values
    df["final_weight"] = w / w.sum()
    df["capped"] = df["final_weight"] >= df["weight_cap"] - 1e-9
    df.attrs.update({"cap_multiple_used": mult, "cap_relaxations": relaxed, "cap_iterations": n_iter,
                     "n_capped": int(df["capped"].sum())})
    return df
