#!/usr/bin/env python
"""Build the full Chinese research report (Markdown + DOCX) from the backtest outputs.

    python scripts/generate_research_report.py            # requires a completed `python main.py` run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import DataHub  # noqa: E402
from src.research_report import build  # noqa: E402
from src.utils import TradingCalendar, load_config, setup_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--latex", action="store_true", help="also write the arXiv-style LaTeX source and compile it to PDF (tectonic / latexmk)")
    args = ap.parse_args()
    setup_logging(args.log_level)
    cfg = load_config(args.config)
    hub = DataHub(cfg)
    idx = hub.index_daily(cfg["data"]["parent_index_code"])
    cal = TradingCalendar(idx["date"])
    out = build(cfg, hub, cal, latex=args.latex)
    for k, v in out.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
