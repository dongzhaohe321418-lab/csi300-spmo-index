#!/usr/bin/env python
"""Step 1 — download / refresh all raw data into data/raw (resumable, cached).

    python scripts/download_data.py            # everything the universe needs
    python scripts/download_data.py --update   # re-download time series (append new days)
    python scripts/download_data.py --sources prices,financials

Sources: index (CSIndex OHLC + current list), universe (CSIndex announcements),
prices (Sina daily bars), financials (Eastmoney income statements),
dividends (Eastmoney), shares (Eastmoney share structure), rights (Eastmoney 配股), holders (Sina top-10 shareholders).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.data import DataHub, NoData  # noqa: E402
from src.universe import build_universe  # noqa: E402
from src.utils import TradingCalendar, ensure_dirs, load_config, load_env, setup_logging  # noqa: E402

log = logging.getLogger("download")


def _run_many(name: str, codes: list[str], fn, workers: int) -> dict:
    ok, nodata, failed = 0, [], []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, c): c for c in codes}
        for i, fut in enumerate(as_completed(futs), 1):
            c = futs[fut]
            try:
                fut.result()
                ok += 1
            except NoData:
                nodata.append(c)
            except Exception as exc:  # noqa: BLE001
                failed.append((c, str(exc)[:120]))
            if i % 50 == 0 or i == len(codes):
                log.info("%s: %d/%d done (%d no-data, %d failed) %.0fs", name, i, len(codes), len(nodata), len(failed), time.time() - t0)
    return {"ok": ok, "nodata": nodata, "failed": failed}


def download_all(hub: DataHub, cal: TradingCalendar, refresh: bool = False,
                 sources: str = "index,universe,prices,financials,dividends,shares,rights", workers: int = 3,
                 reverse: bool = False) -> dict:
    cfg = hub.cfg
    sources = set(sources.split(","))
    # --- index level data & current list ------------------------------------
    idx = hub.index_daily(cfg["data"]["parent_index_code"], refresh=refresh)
    hub.index_daily(cfg["data"]["parent_index_tr_code"], refresh=refresh)
    log.info("CSI300 index history %s -> %s (%d days)", idx["date"].min().date(), idx["date"].max().date(), len(idx))
    hub.csi.current_constituents(refresh=refresh)
    hub.csi.current_weights(refresh=refresh)

    # --- universe -------------------------------------------------------------
    intervals_path = hub.processed / "csi300_membership_intervals.csv"
    if "universe" in sources or not intervals_path.exists():
        intervals, audit, events = build_universe(hub, cal, refresh=refresh)
        log.info("universe: %d securities, %d membership intervals, %d events", intervals["code"].nunique(), len(intervals), len(events))
    intervals = pd.read_csv(intervals_path, dtype={"code": str})
    codes = sorted(set(intervals["code"]), reverse=reverse)
    log.info("%d securities ever in the CSI 300", len(codes))

    summary = {}
    if "prices" in sources:
        # Sina's JS decoder is not thread-safe and the site rate-limits aggressively -> 1 worker
        summary["prices"] = _run_many("prices(sina)", codes, lambda c: hub.stock_daily(c, refresh=refresh), workers=1)
    if "shares" in sources:
        summary["shares"] = _run_many("shares(eastmoney)", codes, lambda c: hub.shares(c, refresh=refresh), workers=workers)
    if "dividends" in sources:
        summary["dividends"] = _run_many("dividends(eastmoney)", codes, lambda c: hub.dividends(c, refresh=refresh), workers=workers)
    if "rights" in sources:
        hub.rights_issues(refresh=refresh)
    if "holders" in sources:
        summary["holders"] = _run_many("holders(sina)", codes, lambda c: hub.holders(c, refresh=refresh), workers=min(workers, 3))
    if "financials" in sources:
        summary["financials"] = _run_many("financials(eastmoney)", codes, lambda c: hub.income(c, refresh=refresh), workers=workers)

    for k, v in summary.items():
        log.info("%s: ok=%d nodata=%d failed=%d", k, v["ok"], len(v["nodata"]), len(v["failed"]))
        if v["nodata"]:
            log.info("   no-data codes: %s", ",".join(v["nodata"][:40]) + (" ..." if len(v["nodata"]) > 40 else ""))
        for c, msg in v["failed"][:20]:
            log.warning("   failed %s: %s", c, msg)
    pd.Series({k: len(v["failed"]) for k, v in summary.items()}).to_json(hub.processed / "download_summary.json")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--update", action="store_true", help="refresh cached time series")
    ap.add_argument("--sources", default="index,universe,prices,financials,dividends,shares,rights",
                    help="comma list; add holders to fetch Sina top-10 shareholders (optional free-float refinement)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--reverse", action="store_true", help="process securities in reverse order (run a 2nd instance in parallel)")
    args = ap.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    load_env(cfg)
    ensure_dirs(cfg)
    hub = DataHub(cfg)
    idx = hub.index_daily(cfg["data"]["parent_index_code"], refresh=args.update)
    cal = TradingCalendar(idx["date"])
    download_all(hub, cal, refresh=args.update, sources=args.sources, workers=args.workers, reverse=args.reverse)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
