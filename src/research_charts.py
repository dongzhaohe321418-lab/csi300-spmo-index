"""Figures for the research report (output/charts/research/). English labels; Chinese text lives in the report."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

matplotlib.use("Agg")

C = {"bench": "#2f3e56", "strategy": "#c0392b", "strategy_net": "#e07b6f", "grey": "#7f8c8d", "blue": "#2e6fbd",
     "green": "#2e8b57", "orange": "#e69f00", "purple": "#7b5ea7", "teal": "#1b9e9e", "light": "#bdc3c7"}
SIZES = (9, 8, 7)


def _font_family() -> list[str]:
    """Latin font first, then the first CJK-capable font installed (macOS / Windows / Linux names)."""
    from matplotlib import font_manager as fm

    installed = {f.name for f in fm.fontManager.ttflist}
    fam = [f for f in ("Helvetica", "Arial") if f in installed][:1]
    for cjk in ("PingFang SC", "Heiti SC", "Hiragino Sans GB", "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
                "WenQuanYi Zen Hei", "Arial Unicode MS"):
        if cjk in installed:
            fam.append(cjk)
            break
    return fam + ["DejaVu Sans"]


def _style() -> None:
    plt.rcParams.update({
        "font.size": SIZES[0], "axes.titlesize": SIZES[0], "axes.labelsize": SIZES[0], "legend.fontsize": SIZES[1],
        "xtick.labelsize": SIZES[2], "ytick.labelsize": SIZES[2], "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlelocation": "left", "axes.titleweight": "normal", "legend.frameon": False, "figure.dpi": 150,
        "savefig.dpi": 200, "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "font.family": _font_family(), "axes.unicode_minus": False,
    })


def _plain_log(ax, ticks) -> None:
    ax.set_yscale("log")
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:,.0f}" if t >= 10 else f"{t:g}" for t in ticks])
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())


def _end_labels(ax, series: dict[str, pd.Series], colors: dict[str, str], fmt="{:,.0f}", xpad_days: int = 60) -> None:
    """Direct labels at the right end of each line, nudged apart vertically (log-aware)."""
    items = sorted(series.items(), key=lambda kv: kv[1].iloc[-1])
    log = ax.get_yscale() == "log"
    ys = [np.log(v.iloc[-1]) if log else v.iloc[-1] for _, v in items]
    lo, hi = ax.get_ylim()
    span = (np.log(hi) - np.log(lo)) if log else (hi - lo)
    min_gap = span * 0.035
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < min_gap:
            ys[i] = ys[i - 1] + min_gap
    x = items[0][1].index[-1] + pd.Timedelta(days=xpad_days)
    for (name, v), y in zip(items, ys):
        yy = np.exp(y) if log else y
        ax.annotate(f"{name}  {fmt.format(v.iloc[-1])}", (x, yy), fontsize=SIZES[1], color=colors[name], va="center", ha="left",
                    annotation_clip=False)


def _year_ticks(ax, start, end, step: int) -> None:
    """Year ticks limited to the data range (the axis itself extends further to hold end-of-line labels)."""
    y0 = pd.Timestamp(start).year + (1 if pd.Timestamp(start).month > 6 else 0)
    years = list(range(y0, pd.Timestamp(end).year + 1, step))
    ax.set_xticks([pd.Timestamp(f"{y}-01-01") for y in years])
    ax.set_xticklabels([str(y) for y in years])


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- figures
def attribution_chart(levels: pd.DataFrame, attr: pd.DataFrame, path: Path) -> Path:
    _style()
    s = pd.DataFrame({
        "CSI 300 (official)": levels["CSI300"],
        "CSI 300 replica, proxy weights": attr["CSI300_replica_proxy"],
        "Selected basket, cap-weighted": attr["Selected_basket_capweighted"],
        "Selected basket, equal-weighted": attr["Selected_basket_equalweighted"],
        "Strategy (SPMO weights)": levels["CSI300_SP_FV_SPMO"],
    }).dropna()
    s = s / s.iloc[0] * 100
    colors = {"CSI 300 (official)": C["bench"], "CSI 300 replica, proxy weights": C["grey"], "Selected basket, cap-weighted": C["blue"],
              "Selected basket, equal-weighted": C["green"], "Strategy (SPMO weights)": C["strategy"]}
    widths = {"Strategy (SPMO weights)": 1.8, "CSI 300 (official)": 1.6}
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for col in s.columns:
        ax.plot(s.index, s[col], color=colors[col], lw=widths.get(col, 1.0), alpha=1 if col in widths else 0.9)
    _plain_log(ax, [50, 100, 200, 300, 500, 700, 1000])
    ax.set_xlim(s.index[0], s.index[-1] + pd.Timedelta(days=365 * 4.2))
    ax.set_ylim(60, max(1100, s.max().max() * 1.15))
    _end_labels(ax, {c: s[c] for c in s.columns}, colors)
    ax.set_title("Where the excess return comes from: same selected baskets under three weighting schemes vs the parent index (¥100, log scale)")
    ax.set_ylabel("Growth of ¥100 (log)")
    _year_ticks(ax, s.index[0], s.index[-1], 2)
    return _save(fig, path)


def cycle_cagr_chart(cycles: pd.DataFrame, path: Path) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(cycles))
    w = 0.38
    b1 = ax.bar(x - w / 2, cycles["CSI300_CAGR"] * 100, w, color=C["bench"], label="CSI 300")
    b2 = ax.bar(x + w / 2, cycles["Strategy_CAGR"] * 100, w, color=C["strategy"], label="Strategy")
    for i, (a, b) in enumerate(zip(cycles["CSI300_CAGR"], cycles["Strategy_CAGR"])):
        top = max(a, b, 0) * 100
        ax.annotate(f"{(b - a) * 100:+.1f} pp", (x[i], top + 3), ha="center", fontsize=SIZES[1],
                    color=C["strategy"] if b > a else C["bench"])
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in cycles["Cycle"]])
    ax.set_ylabel("Annualised return (%)")
    ax.set_title("Strategy vs CSI 300 by market cycle: 2005-2016 flat to negative, the excess return is concentrated in 2017-2020 (labels = strategy − CSI 300, pp of annualised return)")
    ax.legend(loc="upper right", ncol=2)
    ax.margins(y=0.15)
    return _save(fig, path)


def rolling_chart(rbt: pd.DataFrame, path: Path) -> Path:
    _style()
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True, gridspec_kw={"hspace": 0.12})
    axes[0].plot(rbt.index, rbt["beta"], color=C["strategy"], lw=1)
    axes[0].axhline(1.0, color=C["grey"], lw=0.8, ls="--")
    axes[0].set_ylabel("Beta to CSI 300 (1y)")
    axes[0].set_title("Rolling one-year beta stays near 1, tracking error 6-18%, and the excess return alternates in multi-year waves")
    axes[1].plot(rbt.index, rbt["tracking_error"] * 100, color=C["blue"], lw=1)
    axes[1].set_ylabel("Tracking error (1y, %)")
    ex = rbt["rolling_excess_1y"] * 100
    axes[2].fill_between(ex.index, ex, 0, where=ex >= 0, color=C["strategy"], alpha=0.5, lw=0)
    axes[2].fill_between(ex.index, ex, 0, where=ex < 0, color=C["bench"], alpha=0.5, lw=0)
    axes[2].axhline(0, color="black", lw=0.6)
    axes[2].set_ylabel("1y excess return (pp)")
    axes[2].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for ax in axes:
        ax.margins(x=0.01)
    return _save(fig, path)


def monthly_excess_chart(ex: pd.Series, path: Path) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(7.5, 4))
    v = ex.values * 100
    bins = np.arange(np.floor(v.min() / 2) * 2, np.ceil(v.max() / 2) * 2 + 2, 2)
    ax.hist(v, bins=bins, color=C["strategy"], alpha=0.75, edgecolor="white")
    ax.axvline(v.mean(), color="black", lw=1)
    ax.annotate(f"mean {v.mean():+.2f} pp/month\nmedian {np.median(v):+.2f}\nshare > 0: {np.mean(v > 0):.0%}\nn = {len(v)} months",
                (0.98, 0.95), xycoords="axes fraction", ha="right", va="top", fontsize=SIZES[1])
    ax.set_xlabel("Monthly return, strategy − CSI 300 (pp)")
    ax.set_ylabel("Months")
    ax.set_title("Monthly excess returns: a small positive centre with fat tails on both sides")
    ax.margins(x=0.02)
    return _save(fig, path)


def concentration_chart(hc: pd.DataFrame, path: Path) -> Path:
    _style()
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"hspace": 0.12})
    d = hc["effective_date"]
    axes[0].plot(d, hc["n"], color=C["bench"], lw=1.4, marker="o", ms=2.5, label="Constituents")
    axes[0].plot(d, hc["effective_n"], color=C["strategy"], lw=1.4, marker="o", ms=2.5, label="Effective N (1/Σw²)")
    axes[0].set_ylabel("Number of securities")
    axes[0].set_title("Portfolio breadth and concentration per rebalance: ~50 names but an effective N of only 20-37")
    axes[0].legend(loc="center right")
    axes[1].plot(d, hc["top10_weight"] * 100, color=C["strategy"], lw=1.4, marker="o", ms=2.5, label="Top-10 weight")
    axes[1].plot(d, hc["max_weight"] * 100, color=C["orange"], lw=1.2, marker="o", ms=2.5, label="Largest weight (cap 9%)")
    axes[1].plot(d, hc["csi300_weight_covered"] * 100, color=C["grey"], lw=1.2, marker="o", ms=2.5, label="Share of CSI 300 cap held (proxy)")
    axes[1].set_ylabel("Weight (%)")
    axes[1].legend(loc="upper right", ncol=3)
    axes[1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for ax in axes:
        ax.margins(x=0.02, y=0.12)
    return _save(fig, path)


def funnel_chart(fun: pd.DataFrame, path: Path) -> Path:
    _style()
    fail_cols = [c for c in fun.columns if c.startswith("fail:")]
    groups = {
        "No statements available (mostly delisted)": [c for c in fail_cols if "no financial statements" in c or "no report public" in c],
        "Latest quarter ≤ 0 only": [c for c in fail_cols if c == "fail:latest quarter <= 0"],
        "Trailing 4Q sum ≤ 0 only": [c for c in fail_cols if c == "fail:trailing 4Q sum <= 0"],
        "Both tests fail": [c for c in fail_cols if "latest quarter <= 0; trailing" in c],
        "< 4 consecutive quarters public": [c for c in fail_cols if "consecutive" in c],
    }
    data = pd.DataFrame({k: (fun[v].sum(axis=1) if v else pd.Series(0.0, index=fun.index)) for k, v in groups.items()})
    data.index = pd.to_datetime(fun["reference_date"]).values
    colors = [C["grey"], C["orange"], C["blue"], C["strategy"], C["purple"]]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    bottom = np.zeros(len(data))
    x = np.arange(len(data))
    for (k, v), col in zip(data.items(), colors):
        ax.bar(x, v.values, bottom=bottom, color=col, width=0.8, label=k)
        bottom += v.values
    ax.set_xticks(x[::4])
    ax.set_xticklabels([d.strftime("%Y-%m") for d in data.index[::4]], rotation=0)
    ax.set_ylabel("CSI 300 members failing Financial Viability")
    ax.set_title("Financial Viability removes 19-76 of the 300 members per review; loss-makers dominate, missing statements matter early on")
    ax.legend(loc="upper right", ncol=2)
    ax.margins(y=0.15)
    return _save(fig, path)


def persistence_chart(pers: dict, path: Path) -> Path:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.35})
    top = pers["top"].head(20).iloc[::-1]
    axes[0].barh(np.arange(len(top)), top["periods_selected"], color=C["strategy"])
    axes[0].set_yticks(np.arange(len(top)))
    axes[0].set_yticklabels([f"{n} ({t})" for n, t in zip(top["name"].str.replace(" ", ""), top["ticker"])])
    axes[0].set_xlabel(f"Rebalances selected (of {pers['n_periods']})")
    axes[0].set_title("Most persistent constituents")
    axes[0].grid(axis="y", visible=False)
    sp = pers["spells"]
    vc = sp.value_counts().sort_index()
    axes[1].bar(vc.index, vc.values, color=C["bench"])
    axes[1].set_xlabel("Length of an uninterrupted membership spell (rebalances)")
    axes[1].set_ylabel("Number of spells")
    axes[1].set_title(f"Half of all spells last one rebalance (mean {pers['mean_spell']:.1f})")
    axes[1].set_xticks(vc.index)
    return _save(fig, path)


def sector_chart(sx: pd.DataFrame, path: Path) -> Path:
    _style()
    cols = list(sx.columns)
    colors = [C["bench"], C["strategy"], C["strategy_net"]]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    y = np.arange(len(sx))
    h = 0.8 / len(cols)
    for i, (col, cc) in enumerate(zip(cols, colors)):
        ax.barh(y + (i - (len(cols) - 1) / 2) * h, sx[col] * 100, h, color=cc, label=col)
    ax.set_yticks(y)
    ax.set_yticklabels(sx.index)
    ax.invert_yaxis()
    ax.set_xlabel("Weight (%)")
    ax.set_title("Sector exposure (CSI level-1 classification, current): the strategy is a concentrated IT / materials / telecom tilt")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def hsmo_chart(levels: pd.DataFrame, reps: dict[str, pd.Series], path: Path, window_start: str = "2019-01-02", window_end: str = "2025-12-19") -> Path:
    """Two panels: the HSI300-Momentum-Strategy window and the full history, identical data."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), gridspec_kw={"wspace": 0.28})
    series_full = {
        "CSI 300 (price)": levels["CSI300"], "CSI 300 total return": levels["CSI300_TR"],
        "Our strategy, TR net of costs": levels["CSI300_SP_FV_SPMO_TR_Net"],
        "HSMO rules, 15 bp costs": reps["full_nocap"], "HSMO rules + sector cap (current map)": reps["full_cap"],
    }
    colors = {"CSI 300 (price)": C["bench"], "CSI 300 total return": C["grey"], "Our strategy, TR net of costs": C["strategy"],
              "HSMO rules, 15 bp costs": C["blue"], "HSMO rules + sector cap (current map)": C["teal"]}
    for ax, (start, end, title, ticks) in zip(axes, [
            (window_start, window_end, f"HSMO's own window {window_start[:4]}-{window_end[:4]}", [80, 100, 150, 200, 300, 400]),
            (levels.index[0], levels.index[-1], "Full history 2005-2026 (same rules, same data)", [50, 100, 200, 400, 700, 1000, 1500, 2500])]):
        sub = {}
        for k, v in series_full.items():
            if ax is axes[0]:
                v = {"HSMO rules, 15 bp costs": reps["win_nocap"], "HSMO rules + sector cap (current map)": reps["win_cap"]}.get(k, v)
            vv = v.loc[pd.Timestamp(start):pd.Timestamp(end)].dropna()
            # window panel: rebase everything to 100; full panel: only index point series (the strategy /
            # HSMO series already start at ¥100 net of the initial purchase cost)
            sub[k] = vv / vv.iloc[0] * 100 if (ax is axes[0] or abs(vv.iloc[0] - 100) > 5) else vv
        for k, v in sub.items():
            ax.plot(v.index, v, color=colors[k], lw=1.7 if k == "Our strategy, TR net of costs" else 1.0)
        _plain_log(ax, ticks)
        ax.set_ylim(min(v.min() for v in sub.values()) * 0.9, max(v.max() for v in sub.values()) * 1.15)
        span_days = (sub["CSI 300 (price)"].index[-1] - sub["CSI 300 (price)"].index[0]).days
        ax.set_xlim(sub["CSI 300 (price)"].index[0], sub["CSI 300 (price)"].index[-1] + pd.Timedelta(days=span_days * 0.42))
        _end_labels(ax, sub, colors, xpad_days=int(span_days * 0.012))
        ax.set_title(title)
        ax.set_ylabel("Growth of ¥100 (log)")
        _year_ticks(ax, sub["CSI 300 (price)"].index[0], sub["CSI 300 (price)"].index[-1], 1 if ax is axes[0] else 3)
    fig.suptitle("Top-20 equal-weight quarterly momentum (HSMO rules) vs this index on identical data: higher return, but far more concentrated and about twice the turnover",
                 x=0.01, ha="left", fontsize=SIZES[0])
    return _save(fig, path)


def drawdown_compare_chart(levels: pd.DataFrame, hsmo_full: pd.Series, path: Path) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for name, lv, col in (("CSI 300", levels["CSI300"], C["bench"]), ("Our strategy (PR)", levels["CSI300_SP_FV_SPMO"], C["strategy"]),
                          ("HSMO rules (TR, net)", hsmo_full, C["blue"])):
        dd = (lv / lv.cummax() - 1) * 100
        ax.plot(dd.index, dd, color=col, lw=0.9, label=f"{name} (min {dd.min():.0f}%)")
    ax.set_ylabel("Drawdown from peak (%)")
    ax.set_title("Drawdowns: all three approaches lose ~70-77% in 2008; the concentrated HSMO portfolio recovers faster after 2015")
    ax.legend(loc="lower right", ncol=3)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.margins(x=0.01)
    return _save(fig, path)
