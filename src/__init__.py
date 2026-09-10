"""CSI300 S&P Financial Viability + SPMO Momentum Index — research package.

Pipeline (fixed, see README):
    Historical CSI300 (point-in-time)  ->  S&P Financial Viability
    ->  SPMO risk-adjusted momentum    ->  Top 20% (with S&P buffer)
    ->  SPMO weighting (FMC x score, company cap)  ->  Index  ->  Backtest vs CSI300
"""

__version__ = "1.0.0"
