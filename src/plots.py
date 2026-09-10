"""Charts: growth of ¥100, candlesticks (D/W/M, with moving averages), relative strength,
drawdowns, annual & rolling returns, turnover.  Candles use the A-share colour convention
(red = up, green = down).  Moving averages are chart overlays only — they play no role in
selection or weighting."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.ticker as mticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from . import metrics as M  # noqa: E402

COLORS = {"strategy": "#c0392b", "benchmark": "#2c3e50", "up": "#d62728", "down": "#2ca02c",
          "ma20": "#ff7f0e", "ma50": "#1f77b4", "ma100": "#9467bd", "ma200": "#8c564b"}
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 130,
                     "font.sans-serif": ["Arial Unicode MS", "PingFang SC", "Heiti SC", "SimHei", "DejaVu Sans"],
                     "axes.unicode_minus": False})


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# OHLC resampling & moving averages
# --------------------------------------------------------------------------- #
def resample_ohlc(ohlc: pd.DataFrame, freq: str) -> pd.DataFrame:
    """freq: 'D' (unchanged), 'W' (weekly, calendar weeks ending Friday), 'M' (calendar months).
    Open = first bar's Open, High = max High, Low = min Low, Close = last bar's Close."""
    if freq.upper() == "D":
        return ohlc.copy()
    rule = {"W": "W-FRI", "M": "ME"}[freq.upper()]
    try:
        g = ohlc.resample(rule)
    except ValueError:  # older pandas
        g = ohlc.resample({"W": "W-FRI", "M": "M"}[freq.upper()])
    out = pd.DataFrame({"Open": g["Open"].first(), "High": g["High"].max(), "Low": g["Low"].min(), "Close": g["Close"].last()})
    out = out.dropna(subset=["Close"])
    # label each bar by the last trading day inside the period
    last_day = ohlc["Close"].groupby(ohlc.index.to_period({"W": "W-FRI", "M": "M"}[freq.upper()])).apply(lambda s: s.index[-1])
    out.index = pd.DatetimeIndex(last_day.values)
    out.index.name = "Date"
    return out


def moving_averages(close_daily: pd.Series, windows=(20, 50, 100, 200)) -> pd.DataFrame:
    return pd.DataFrame({f"MA{w}": close_daily.rolling(w).mean() for w in windows})


def draw_candles(ax, ohlc: pd.DataFrame, title: str, ma: Optional[pd.DataFrame] = None, width: float = 0.6) -> None:
    x = np.arange(len(ohlc))
    o, h, l, c = ohlc["Open"].values, ohlc["High"].values, ohlc["Low"].values, ohlc["Close"].values
    up = c >= o
    for i in range(len(ohlc)):
        color = COLORS["up"] if up[i] else COLORS["down"]
        ax.plot([x[i], x[i]], [l[i], h[i]], color=color, linewidth=0.8, zorder=2)
        body_low, body_h = min(o[i], c[i]), abs(c[i] - o[i])
        ax.add_patch(Rectangle((x[i] - width / 2, body_low), width, max(body_h, (h.max() - l.min()) * 1e-4),
                               facecolor=color if not up[i] else "white", edgecolor=color, linewidth=0.8, zorder=3))
    if ma is not None:
        for col in ma.columns:
            ax.plot(x, ma[col].reindex(ohlc.index).values, linewidth=1.2, label=col, color=COLORS.get(col.lower(), None), zorder=4)
        ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(-1, len(ohlc))
    ax.set_title(title)
    # year ticks (first bar of each calendar year; skip a partial first year to avoid overlaps)
    years = ohlc.index.year
    ticks = [i for i in range(1, len(ohlc)) if years[i] != years[i - 1]]
    if not ticks or ticks[0] > len(ohlc) * 0.08:
        ticks = [0] + ticks
    if len(ticks) > 30:
        ticks = ticks[::2]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(ohlc.index[i].year) for i in ticks], rotation=0)
    ax.set_ylabel("Index level")


def _plain_log_axis(ax, minor=(1500, 2000, 3000, 5000, 7000), fmt="{:,.0f}") -> None:
    """Log y-axis with plain numbers (1000, 2000, ...) instead of 10^3 notation."""
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: fmt.format(v)))
    ax.yaxis.set_minor_formatter(mticker.FuncFormatter(lambda v, _: fmt.format(v) if any(abs(v - m) < 1e-9 * max(1.0, abs(m)) for m in minor) else ""))


def candlestick_chart(ohlc: pd.DataFrame, title: str, path: Path, ma: Optional[pd.DataFrame] = None, log_scale: bool = False) -> Path:
    fig, ax = plt.subplots(figsize=(13, 5.5))
    draw_candles(ax, ohlc, title, ma)
    if log_scale:
        _plain_log_axis(ax)
    return _save(fig, path)


def candlestick_comparison(strat: pd.DataFrame, bench: pd.DataFrame, path: Path, freq_label: str,
                           ma_s: Optional[pd.DataFrame] = None, ma_b: Optional[pd.DataFrame] = None) -> Path:
    common = strat.index.intersection(bench.index)
    s, b = strat.loc[common], bench.loc[common]
    fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
    draw_candles(axes[0], s, f"CSI300 S&P Financial Viability + SPMO ({freq_label} candles, log scale)", ma_s)
    draw_candles(axes[1], b, f"CSI 300 ({freq_label} candles, log scale)", ma_b)
    for ax in axes:
        _plain_log_axis(ax)
    axes[1].set_xlabel(f"{common[0].date()} → {common[-1].date()}  (identical date range, aligned {freq_label.lower()} bars)")
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# Line charts
# --------------------------------------------------------------------------- #
def growth_of_100(levels: pd.DataFrame, path: Path, log_scale: bool = False, net: Optional[pd.Series] = None) -> Path:
    g = 100.0 * levels / levels.iloc[0]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(g.index, g["CSI300"], color=COLORS["benchmark"], label=f"CSI 300  (final ¥{g['CSI300'].iloc[-1]:,.0f})")
    ax.plot(g.index, g["CSI300_SP_FV_SPMO"], color=COLORS["strategy"],
            label=f"CSI300 S&P Financial Viability + SPMO  (final ¥{g['CSI300_SP_FV_SPMO'].iloc[-1]:,.0f})")
    if net is not None:
        gn = 100.0 * net / net.iloc[0]
        ax.plot(gn.index, gn.values, color=COLORS["strategy"], linestyle="--", linewidth=1, alpha=0.8,
                label=f"Strategy net of costs  (final ¥{gn.iloc[-1]:,.0f})")
    ax.set_ylabel("Growth of ¥100" + (" (log scale)" if log_scale else ""))
    ax.set_xlabel("Date")
    if log_scale:
        _plain_log_axis(ax, minor=(150, 200, 300, 400, 500, 600, 700, 800, 900, 1500, 2000))
    ax.set_title("Growth of ¥100: CSI 300 vs CSI300 S&P Financial Viability + SPMO")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    return _save(fig, path)


def relative_strength(rs: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5))
    ratio = rs["Strategy_to_CSI300_Ratio"]
    ax.plot(ratio.index, ratio, color=COLORS["strategy"])
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1)
    ax.set_title("Relative Strength: Strategy / CSI 300 (rising = strategy outperforming)")
    ax.set_ylabel("Ratio (start = 1.0)")
    _plain_log_axis(ax, minor=(), fmt="{:g}")
    lo, hi = float(ratio.min()), float(ratio.max())
    ticks = [t for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0) if lo * 0.97 <= t <= hi * 1.03]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    return _save(fig, path)


def drawdowns(levels: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5))
    for col, label, color in (("CSI300", "CSI 300", COLORS["benchmark"]), ("CSI300_SP_FV_SPMO", "Strategy", COLORS["strategy"])):
        dd = M.drawdown_series(levels[col])
        ax.fill_between(dd.index, dd.values, 0, alpha=0.25, color=color)
        ax.plot(dd.index, dd.values, color=color, linewidth=0.9, label=f"{label} (max {dd.min():.1%})")
    ax.set_title("Drawdown comparison")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(loc="lower left")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    return _save(fig, path)


def annual_returns_bar(ann: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(ann))
    w = 0.4
    ax.bar(x - w / 2, ann["CSI300"], width=w, color=COLORS["benchmark"], label="CSI 300")
    ax.bar(x + w / 2, ann["CSI300_SP_FV_SPMO"], width=w, color=COLORS["strategy"], label="Strategy")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in ann.index], rotation=45)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Calendar-year returns")
    ax.legend()
    return _save(fig, path)


def rolling_chart(series: dict[str, pd.Series], title: str, ylabel: str, path: Path, zero_line: bool = True) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5))
    for label, s in series.items():
        color = COLORS["strategy"] if "Strategy" in label or "Excess" in label else COLORS["benchmark"]
        if "3Y" in label:
            color = "#e67e22"
        if "5Y" in label:
            color = "#8e44ad"
        ax.plot(s.index, s.values, label=label, color=color, linewidth=1.1)
    if zero_line:
        ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    return _save(fig, path)


def turnover_chart(turnover: pd.DataFrame, path: Path) -> Path:
    t = turnover.iloc[1:]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(pd.to_datetime(t["effective_date"]), t["one_way_turnover"], width=60, color=COLORS["strategy"])
    ax.axhline(t["one_way_turnover"].mean(), color="black", linestyle="--", linewidth=1,
               label=f"mean per rebalance {t['one_way_turnover'].mean():.1%}")
    ax.set_title("One-way turnover at each semi-annual rebalance")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend()
    return _save(fig, path)
