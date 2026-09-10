#!/usr/bin/env python
"""Step 4 — Markdown research report (output/report/backtest_report.md) and README results block."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import Context  # noqa: E402
from src.utils import setup_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    setup_logging()
    ctx = Context(args.config)
    intervals = ctx.load_intervals()
    bt = ctx.load_backtest()
    an = ctx.stage_analysis(bt)
    ctx.stage_report(bt, an, intervals)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
