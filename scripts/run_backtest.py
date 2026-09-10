#!/usr/bin/env python
"""Step 3 — performance statistics, CSV outputs and charts from the engine results
(re-runs the score/backtest stage if its outputs are missing)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import Context  # noqa: E402
from src.utils import abs_path, setup_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    setup_logging()
    ctx = Context(args.config)
    perf = abs_path(ctx.cfg, ctx.cfg["output"]["performance_dir"])
    if not (perf / "_engine_strategy_pr.csv").exists():
        intervals = ctx.load_intervals()
        ctx.stage_prices(intervals)
        bt = ctx.stage_backtest(intervals, ctx.load_panel())
    else:
        bt = ctx.load_backtest()
    ctx.stage_analysis(bt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
