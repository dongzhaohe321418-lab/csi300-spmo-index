"""Optional Tushare Pro provider (data.provider: tushare, token in .env as TUSHARE_TOKEN).

It returns frames with the SAME schemas as the free clients in `data.py` so the rest of the
pipeline is source-agnostic:

    stock_daily(code)  -> code, date, prevclose, open, high, low, close, volume(shares), amount(CNY)
    income(code)       -> code, REPORT_DATE, NOTICE_DATE, NETPROFIT, CONTINUED_NETPROFIT, PARENT_NETPROFIT
    dividends(code)    -> code, report_period, cash_div_per10, bonus_per10, ex_date, record_date, status
    shares(code)       -> code, change_date, total_shares, float_shares, listed_a_shares, reason

Not exercised by the default (free-data) run; requires a Tushare account with >= 2000 points.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pandas as pd

from .data import Cache, NoData
from .utils import em_secucode

log = logging.getLogger(__name__)


class TushareProvider:
    def __init__(self, cache: Cache, cfg: dict):
        token = os.environ.get("TUSHARE_TOKEN", "")
        if not token:
            raise RuntimeError("data.provider is 'tushare' but TUSHARE_TOKEN is not set (put it in .env)")
        import tushare as ts

        ts.set_token(token)
        self.pro = ts.pro_api()
        self.cache = cache
        self.sleep_s = float(cfg["data"].get("request_sleep_s", 0.35))

    def _cached(self, key: str, fetch, refresh: bool = False) -> pd.DataFrame:
        p = self.cache.path("tushare", f"{key}.parquet")
        if p.exists() and not refresh:
            return self.cache.read_df(p)
        time.sleep(self.sleep_s)
        df = fetch()
        if df is None or df.empty:
            raise NoData(key)
        self.cache.write_df(df, p)
        return df

    def stock_daily(self, code: str, refresh: bool = False) -> pd.DataFrame:
        ts_code = em_secucode(code)

        def fetch():
            frames = []
            for start, end in (("19900101", "20091231"), ("20100101", "20191231"), ("20200101", "20291231")):
                frames.append(self.pro.daily(ts_code=ts_code, start_date=start, end_date=end))
                time.sleep(self.sleep_s)
            df = pd.concat(frames, ignore_index=True)
            if df.empty:
                return df
            out = pd.DataFrame({
                "code": str(code).zfill(6), "date": pd.to_datetime(df["trade_date"]),
                "prevclose": pd.to_numeric(df["pre_close"], errors="coerce"),   # Tushare pre_close is the exchange reference price
                "open": df["open"], "high": df["high"], "low": df["low"], "close": df["close"],
                "volume": df["vol"] * 100.0, "amount": df["amount"] * 1000.0,
            })
            # keep prevclose only where it differs from the previous close (ex-date reference), as with Sina
            out = out.sort_values("date").reset_index(drop=True)
            prev = out["close"].shift(1)
            out.loc[(out["prevclose"] - prev).abs() < 1e-9, "prevclose"] = float("nan")
            return out

        return self._cached(f"daily/{code}", fetch, refresh)

    def income(self, code: str, refresh: bool = False) -> pd.DataFrame:
        ts_code = em_secucode(code)

        def fetch():
            df = self.pro.income(ts_code=ts_code, report_type="1",
                                 fields="ts_code,ann_date,f_ann_date,end_date,n_income,n_income_attr_p,continued_net_profit")
            if df is None or df.empty:
                return df
            out = pd.DataFrame({
                "code": str(code).zfill(6),
                "REPORT_DATE": pd.to_datetime(df["end_date"]),
                "NOTICE_DATE": pd.to_datetime(df["ann_date"]),
                "NETPROFIT": pd.to_numeric(df["n_income"], errors="coerce"),
                "CONTINUED_NETPROFIT": pd.to_numeric(df["continued_net_profit"], errors="coerce"),
                "PARENT_NETPROFIT": pd.to_numeric(df["n_income_attr_p"], errors="coerce"),
            })
            return out.sort_values("REPORT_DATE").drop_duplicates("REPORT_DATE", keep="first").reset_index(drop=True)

        return self._cached(f"income/{code}", fetch, refresh)

    def dividends(self, code: str, refresh: bool = False) -> pd.DataFrame:
        ts_code = em_secucode(code)

        def fetch():
            df = self.pro.dividend(ts_code=ts_code)
            if df is None or df.empty:
                return df
            df = df[df["div_proc"] == "实施"]
            out = pd.DataFrame({
                "code": str(code).zfill(6),
                "report_period": pd.to_datetime(df["end_date"], errors="coerce"),
                "cash_div_per10": pd.to_numeric(df["cash_div_tax"], errors="coerce") * 10.0,
                "bonus_per10": (pd.to_numeric(df["stk_bo_rate"], errors="coerce").fillna(0) + pd.to_numeric(df["stk_co_rate"], errors="coerce").fillna(0)) * 10.0,
                "ex_date": pd.to_datetime(df["ex_date"], errors="coerce"),
                "record_date": pd.to_datetime(df["record_date"], errors="coerce"),
                "status": "实施分配",
            })
            return out.sort_values("ex_date").reset_index(drop=True)

        return self._cached(f"dividend/{code}", fetch, refresh)

    def shares(self, code: str, refresh: bool = False) -> pd.DataFrame:
        ts_code = em_secucode(code)

        def fetch():
            frames = []
            for start, end in (("19900101", "20091231"), ("20100101", "20191231"), ("20200101", "20291231")):
                frames.append(self.pro.daily_basic(ts_code=ts_code, start_date=start, end_date=end, fields="trade_date,total_share,float_share"))
                time.sleep(self.sleep_s)
            df = pd.concat(frames, ignore_index=True)
            if df.empty:
                return df
            df = df.sort_values("trade_date")
            chg = df[(df["float_share"] != df["float_share"].shift(1)) | (df["total_share"] != df["total_share"].shift(1))]
            return pd.DataFrame({
                "code": str(code).zfill(6), "change_date": pd.to_datetime(chg["trade_date"]),
                "total_shares": chg["total_share"] * 1e4, "float_shares": chg["float_share"] * 1e4,
                "listed_a_shares": chg["float_share"] * 1e4, "reason": "daily_basic",
            }).reset_index(drop=True)

        return self._cached(f"shares/{code}", fetch, refresh)
