"""Shared helpers: configuration, logging, paths, trading-calendar utilities."""
from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def setup_logging(level: str = "INFO") -> None:
    os.environ.setdefault("TQDM_DISABLE", "1")   # akshare prints tqdm bars for every request
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("urllib3", "requests", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def load_config(path: str | Path = "config.yaml") -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    with open(p, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_root"] = str(PROJECT_ROOT)
    return cfg


def abs_path(cfg: dict, rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else Path(cfg["_root"]) / p


def ensure_dirs(cfg: dict) -> None:
    for key in ("raw_dir", "processed_dir", "cache_dir"):
        abs_path(cfg, cfg["data"][key]).mkdir(parents=True, exist_ok=True)
    for key, rel in cfg["output"].items():
        abs_path(cfg, rel).mkdir(parents=True, exist_ok=True)


def load_env(cfg: dict) -> None:
    """Load .env (API keys) into os.environ without committing secrets."""
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(cfg["_root"]) / ".env")
    except Exception:  # pragma: no cover - optional dependency
        pass


# --------------------------------------------------------------------------- #
# Code / symbol helpers
# --------------------------------------------------------------------------- #
def exchange_of(code: str) -> str:
    """Return 'SH' or 'SZ' (or 'BJ') for a 6-digit A-share code."""
    c = str(code).zfill(6)
    if c[0] == "6":
        return "SH"
    if c[0] in ("0", "3"):
        return "SZ"
    if c[0] in ("4", "8", "9"):
        return "BJ"
    raise ValueError(f"unknown exchange for code {code}")


def sina_symbol(code: str) -> str:
    return exchange_of(code).lower() + str(code).zfill(6)


def em_symbol(code: str) -> str:
    return exchange_of(code) + str(code).zfill(6)


def em_secucode(code: str) -> str:
    return f"{str(code).zfill(6)}.{exchange_of(code)}"


# --------------------------------------------------------------------------- #
# Calendar helpers (built from the CSI 300 index history = actual A-share days)
# --------------------------------------------------------------------------- #
class TradingCalendar:
    def __init__(self, days: Iterable[pd.Timestamp]):
        self.days = pd.DatetimeIndex(sorted(set(pd.to_datetime(list(days))))).normalize()
        if len(self.days) == 0:
            raise ValueError("empty trading calendar")

    def is_trading_day(self, d) -> bool:
        return pd.Timestamp(d).normalize() in self.days

    def last_on_or_before(self, d) -> pd.Timestamp:
        d = pd.Timestamp(d).normalize()
        pos = self.days.searchsorted(d, side="right") - 1
        if pos < 0:
            raise ValueError(f"no trading day on/before {d.date()}")
        return self.days[pos]

    def first_on_or_after(self, d) -> pd.Timestamp:
        d = pd.Timestamp(d).normalize()
        pos = self.days.searchsorted(d, side="left")
        if pos >= len(self.days):
            raise ValueError(f"no trading day on/after {d.date()}")
        return self.days[pos]

    def next(self, d, n: int = 1) -> pd.Timestamp:
        pos = self.days.searchsorted(pd.Timestamp(d).normalize(), side="right") - 1 + n
        if pos < 0:
            raise ValueError(f"no trading day {n} before {pd.Timestamp(d).date()}")
        if pos >= len(self.days):
            # beyond the known calendar: extrapolate with weekdays (used only for
            # not-yet-effective announcements)
            extra = pos - len(self.days) + 1
            return (self.days[-1] + pd.offsets.BDay(extra)).normalize()
        return self.days[pos]

    def prev(self, d, n: int = 1) -> pd.Timestamp:
        return self.next(d, -n)

    def between(self, start, end) -> pd.DatetimeIndex:
        s = pd.Timestamp(start).normalize()
        e = pd.Timestamp(end).normalize()
        return self.days[(self.days >= s) & (self.days <= e)]

    def month_end(self, year: int, month: int) -> pd.Timestamp:
        """Last trading day of a calendar month."""
        last_cal = (pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)).normalize()
        return self.last_on_or_before(last_cal)

    def third_friday(self, year: int, month: int) -> pd.Timestamp:
        """3rd Friday of the month, rolled back to the last trading day on/before it."""
        first = date(year, month, 1)
        # weekday(): Monday=0 ... Friday=4
        offset = (4 - first.weekday()) % 7
        tf = pd.Timestamp(first + timedelta(days=offset + 14))
        if tf > self.days[-1]:
            return tf            # beyond the known calendar (not yet implemented rebalance)
        return self.last_on_or_before(tf)


def months_before(d, months: int) -> pd.Timestamp:
    """Calendar month arithmetic that clips to month ends (e.g. Aug-31 -> Jul-31)."""
    d = pd.Timestamp(d)
    y, m = d.year, d.month - months
    while m <= 0:
        m += 12
        y -= 1
    last_day = (pd.Timestamp(year=y, month=m, day=1) + pd.offsets.MonthEnd(0)).day
    return pd.Timestamp(year=y, month=m, day=min(d.day, last_day))


def parse_date(x) -> Optional[pd.Timestamp]:
    if x is None or (isinstance(x, float) and pd.isna(x)) or x == "":
        return None
    try:
        return pd.Timestamp(x).normalize()
    except Exception:
        return None


def today_str() -> str:
    return datetime.today().strftime("%Y%m%d")
