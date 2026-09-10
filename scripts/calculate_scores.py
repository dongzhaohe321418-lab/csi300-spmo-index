#!/usr/bin/env python
"""Step 2 — point-in-time universe, adjusted prices, Financial Viability + momentum scores,
selection and weights for every reference date (writes output/scores and output/holdings),
then runs the index engine and the eligibility audit."""
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
    ap.add_argument("--rebuild-prices", action="store_true")
    args = ap.parse_args()
    setup_logging()
    ctx = Context(args.config)
    intervals = ctx.load_intervals()
    ctx.stage_prices(intervals, rebuild=args.rebuild_prices)
    panel = ctx.load_panel()
    ctx.stage_backtest(intervals, panel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
