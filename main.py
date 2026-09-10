#!/usr/bin/env python
"""One-command run of the whole research pipeline:

    python main.py                 # data (cached) -> universe -> scores -> backtest -> charts -> report
    python main.py --update        # also refresh time series from the free sources
    python main.py --skip-download # offline: use whatever is in data/raw
    python main.py --research      # additionally build the full Chinese research report (md + docx)

See README.md for the methodology and the output layout.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline import run_all  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="CSI300 S&P Financial Viability + SPMO Momentum Index backtest")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--update", action="store_true", help="refresh cached time series before running")
    ap.add_argument("--skip-download", action="store_true", help="do not touch the network; use cached raw data")
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--research", action="store_true", help="also generate output/report/research_report_zh.md/.docx")
    args = ap.parse_args()
    run_all(args.config, refresh=args.update, skip_download=args.skip_download, log_level=args.log_level)
    if args.research:
        from src.data import DataHub
        from src.research_report import build
        from src.utils import TradingCalendar, load_config

        cfg = load_config(args.config)
        hub = DataHub(cfg)
        cal = TradingCalendar(hub.index_daily(cfg["data"]["parent_index_code"])["date"])
        build(cfg, hub, cal, latex=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
