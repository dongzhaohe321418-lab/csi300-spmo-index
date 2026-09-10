"""Data access layer (free public sources, cached locally).

Sources
-------
* Sina Finance      — full-history daily OHLC / volume / amount for every A-share
                      (including delisted ones) plus the exchange reference price
                      (`prevclose`) on ex-rights / ex-dividend dates.
* Eastmoney (F10)   — quarterly income statements with first announcement dates
                      (NOTICE_DATE, NETPROFIT, CONTINUED_NETPROFIT, ...), dividend /
                      bonus-share details, share-structure history (流通A股),
                      market-wide rights-issue records.
* CSIndex (official)— CSI 300 (000300) and CSI 300 Total Return (H00300) daily OHLC,
                      current constituent list and weights, and every historical
                      constituent-adjustment announcement (used to rebuild the
                      point-in-time membership history).
* Tushare (optional)— same information from Tushare Pro when TUSHARE_TOKEN is set
                      (see README; not required for the default run).

Everything is cached under data/raw so that `python main.py` can be re-run
offline; pass refresh=True (main.py --update) to re-download time series.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Callable, Iterable, Optional

import pandas as pd
import requests

from .utils import abs_path, em_secucode, em_symbol, exchange_of, sina_symbol

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #


class Cache:
    """Tiny parquet/json cache rooted at data/raw."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, *parts: str) -> Path:
        p = self.root.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @staticmethod
    def age_days(p: Path) -> float:
        return (time.time() - p.stat().st_mtime) / 86400.0

    def read_df(self, p: Path) -> pd.DataFrame:
        return pd.read_parquet(p)

    def write_df(self, df: pd.DataFrame, p: Path) -> None:
        tmp = p.with_suffix(p.suffix + ".tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, p)

    def read_json(self, p: Path):
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def write_json(self, obj, p: Path) -> None:
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False)
        os.replace(tmp, p)


def _retry(fn: Callable, tries: int = 4, base_sleep: float = 1.5, what: str = "") -> object:
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - we re-raise after retries
            last = exc
            wait = base_sleep * (2**i)
            log.debug("retry %d/%d for %s after error %s (sleep %.1fs)", i + 1, tries, what, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"giving up on {what}: {last}") from last


class NoData(Exception):
    """The source has no record for this security (e.g. delisted long ago)."""


def fetch_html(url: str, timeout: int = 60) -> str:
    """GET a page as text; falls back to the system curl when the Python TLS stack fails
    (observed with some large Sina pages behind proxies)."""
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        if r.encoding is None or r.encoding.lower() in ("iso-8859-1", "ascii"):
            r.encoding = r.apparent_encoding
        return r.text
    except (requests.exceptions.SSLError, requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError):
        import subprocess

        res = subprocess.run(["curl", "-sS", "-L", "-m", str(timeout), url], capture_output=True, timeout=timeout + 10)
        if res.returncode != 0 or not res.stdout:
            raise
        raw = res.stdout
        for enc in ("utf-8", "gb18030"):
            try:
                txt = raw.decode(enc)
            except UnicodeDecodeError:
                continue
            if "\ufffd" not in txt[:5000]:
                return txt
        return raw.decode("gb18030", errors="ignore")


def parse_sina_holders(html: str) -> pd.DataFrame:
    """Parse Sina 股本股东-主要股东 page into rows: period_end, ann_date, rank, holder, shares, pct, share_class."""
    from io import StringIO

    tables = pd.read_html(StringIO(html))
    tbl = None
    for t in tables:
        first = t.iloc[:, 0].astype(str)
        if first.str.startswith("截至日期").any() and t.shape[1] >= 5:
            tbl = t.iloc[:, :5].copy()
            break
    if tbl is None:
        raise ValueError("holder table not found")
    tbl.columns = list(range(5))
    col0 = tbl[0].astype(str)
    starts = tbl[col0.str.startswith("截至日期")].index.tolist() + [len(tbl)]
    rows = []
    for i in range(len(starts) - 1):
        blk = tbl.iloc[starts[i]:starts[i + 1]].dropna(how="all")
        if len(blk) < 6:
            continue
        period = str(blk.iloc[0, 1])
        ann = str(blk.iloc[1, 1])
        data = blk.iloc[5:]
        for _, r in data.iterrows():
            rows.append((period, ann, r[0], r[1], r[2], r[3], r[4]))
    return pd.DataFrame(rows, columns=["period_end", "ann_date", "rank", "holder", "shares", "pct", "share_class"])


# --------------------------------------------------------------------------- #
# Sina Finance — daily bars
# --------------------------------------------------------------------------- #
class SinaClient:
    # NB: the "hisdata_klc2" variant carries the exchange ex-date reference price (prevclose);
    # the plain "hisdata" variant does not.
    HIST_URL = "https://finance.sina.com.cn/realstock/company/{symbol}/hisdata_klc2/klc_kl.js"
    HFQ_URL = "https://finance.sina.com.cn/realstock/company/{symbol}/hfq.js"
    HOLDER_URL = "https://vip.stock.finance.sina.com.cn/corp/go.php/vCI_StockHolder/stockid/{code}.phtml"

    def __init__(self, cache: Cache, sleep_s: float = 0.6, retries: int = 4):
        self.cache = cache
        self.sleep_s = sleep_s
        self.retries = retries
        self._js = None

    def _decoder(self):
        if self._js is None:
            import py_mini_racer
            from akshare.stock.cons import hk_js_decode

            ctx = py_mini_racer.MiniRacer()
            ctx.eval(hk_js_decode)
            self._js = ctx
        return self._js

    def _fetch_daily(self, code: str) -> pd.DataFrame:
        sym = sina_symbol(code)
        r = requests.get(self.HIST_URL.format(symbol=sym), timeout=60)
        r.raise_for_status()
        txt = r.text
        if "=" not in txt:
            raise NoData(code)
        payload = txt.split("=", 1)[1].split(";")[0].replace('"', "")
        if not payload.strip():
            raise NoData(code)
        rows = self._decoder().call("d", payload)
        df = pd.DataFrame(rows)
        if df.empty:
            raise NoData(code)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        for c in ("prevclose", "open", "high", "low", "close", "volume", "amount"):
            if c not in df.columns:
                df[c] = pd.NA
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df[["date", "prevclose", "open", "high", "low", "close", "volume", "amount"]]
        df = df.dropna(subset=["close"]).drop_duplicates("date").sort_values("date").reset_index(drop=True)
        df.insert(0, "code", str(code).zfill(6))
        return df

    def daily(self, code: str, refresh: bool = False, max_age_days: float = 1.0) -> pd.DataFrame:
        """Full daily history: code, date, prevclose(ex-date reference), OHLC, volume(shares), amount(CNY)."""
        code = str(code).zfill(6)
        p = self.cache.path("sina", "daily", f"{code}.parquet")
        marker = p.with_suffix(".nodata")
        if not refresh and marker.exists() and Cache.age_days(marker) < 30:
            raise NoData(code)
        if p.exists() and not (refresh and Cache.age_days(p) > max_age_days):
            return self.cache.read_df(p)
        time.sleep(self.sleep_s)
        try:
            df = _retry(lambda: self._fetch_daily(code), self.retries, what=f"sina daily {code}")
        except RuntimeError as exc:
            if isinstance(exc.__cause__, NoData):
                marker.touch()
                raise NoData(code) from exc
            raise
        self.cache.write_df(df, p)
        return df

    def hfq_factor(self, code: str, refresh: bool = False) -> pd.DataFrame:
        """Sina multiplicative 后复权 factors on event dates (used only as a cross-check)."""
        code = str(code).zfill(6)
        p = self.cache.path("sina", "hfq", f"{code}.parquet")
        if p.exists() and not refresh:
            return self.cache.read_df(p)
        time.sleep(self.sleep_s)

        def _fetch():
            r = requests.get(self.HFQ_URL.format(symbol=sina_symbol(code)), timeout=60)
            r.raise_for_status()
            data = json.loads(re.sub(r"^[^=]*=", "", r.text.split("\n")[0]).rstrip(";"))["data"]
            df = pd.DataFrame(data)
            if df.empty:
                return pd.DataFrame(columns=["date", "hfq_factor"])
            df.columns = ["date", "hfq_factor"]
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["hfq_factor"] = pd.to_numeric(df["hfq_factor"], errors="coerce")
            return df.dropna().sort_values("date").reset_index(drop=True)

        df = _retry(_fetch, self.retries, what=f"sina hfq {code}")
        self.cache.write_df(df, p)
        return df

    # -- top-10 shareholders (主要股东) ---------------------------------------
    def shareholders(self, code: str, refresh: bool = False, max_age_days: float = 7.0) -> pd.DataFrame:
        """Quarterly top-10 shareholder history (Sina 股本股东-主要股东) with announcement dates.

        Columns: code, period_end, ann_date, rank, holder, shares, pct, share_class
        Used to estimate the free float (non-free holders >= 5%: state, founders, strategic).
        """
        code = str(code).zfill(6)
        p = self.cache.path("sina", "holders", f"{code}.parquet")
        marker = p.with_suffix(".nodata")
        cols = ["code", "period_end", "ann_date", "rank", "holder", "shares", "pct", "share_class"]
        if not refresh and marker.exists() and Cache.age_days(marker) < 30:
            return pd.DataFrame(columns=cols)
        if p.exists() and not (refresh and Cache.age_days(p) > max_age_days):
            return self.cache.read_df(p)
        time.sleep(self.sleep_s)

        url = self.HOLDER_URL.format(code=code)

        def _fetch():
            html = fetch_html(url)
            try:
                df = parse_sina_holders(html)
            except (IndexError, KeyError, ValueError) as exc:   # page without the holder table (delisted long ago)
                raise NoData(code) from exc
            if df is None or df.empty:
                raise NoData(code)
            out = pd.DataFrame({
                "code": code,
                "period_end": pd.to_datetime(df["period_end"], errors="coerce"),
                "ann_date": pd.to_datetime(df["ann_date"], errors="coerce"),
                "rank": pd.to_numeric(df["rank"], errors="coerce"),
                "holder": df["holder"].astype(str),
                "shares": pd.to_numeric(df["shares"], errors="coerce"),
                "pct": pd.to_numeric(df["pct"].astype(str).str.strip("↓↑"), errors="coerce"),
                "share_class": df["share_class"].astype(str),
            })
            out.loc[out["ann_date"] <= out["period_end"], "ann_date"] = pd.NaT   # placeholder dates (1900-01-01)
            return out.dropna(subset=["period_end"]).sort_values(["period_end", "rank"]).reset_index(drop=True)

        try:
            df = _retry(_fetch, self.retries, what=f"sina holders {code}")
        except RuntimeError as exc:
            if isinstance(exc.__cause__, NoData):
                marker.touch()
                return pd.DataFrame(columns=cols)
            raise
        self.cache.write_df(df, p)
        return df


# --------------------------------------------------------------------------- #
# Eastmoney — fundamentals / corporate actions / share structure
# --------------------------------------------------------------------------- #
INCOME_COLS = [
    "SECURITY_CODE", "SECURITY_NAME_ABBR", "ORG_TYPE", "REPORT_DATE", "REPORT_TYPE",
    "REPORT_DATE_NAME", "NOTICE_DATE", "UPDATE_DATE", "TOTAL_OPERATE_INCOME", "OPERATE_INCOME",
    "NETPROFIT", "CONTINUED_NETPROFIT", "DISCONTINUED_NETPROFIT", "PARENT_NETPROFIT",
    "DEDUCT_PARENT_NETPROFIT",
]


class EastmoneyClient:
    def __init__(self, cache: Cache, sleep_s: float = 0.35, retries: int = 4):
        self.cache = cache
        self.sleep_s = sleep_s
        self.retries = retries

    # -- income statement ---------------------------------------------------
    def income_statement(self, code: str, refresh: bool = False, max_age_days: float = 7.0) -> pd.DataFrame:
        """Quarterly (cumulative YTD) income statements with first announcement date."""
        import akshare as ak

        code = str(code).zfill(6)
        p = self.cache.path("eastmoney", "income", f"{code}.parquet")
        marker = p.with_suffix(".nodata")
        if not refresh and marker.exists() and Cache.age_days(marker) < 30:
            raise NoData(code)
        if p.exists() and not (refresh and Cache.age_days(p) > max_age_days):
            return self.cache.read_df(p)
        time.sleep(self.sleep_s)

        def _fetch():
            try:
                df = ak.stock_profit_sheet_by_report_em(symbol=em_symbol(code))
            except KeyError as exc:  # Eastmoney returns no "data" for delisted companies
                raise NoData(code) from exc
            if df is None or df.empty:
                raise NoData(code)
            cols = [c for c in INCOME_COLS if c in df.columns]
            out = df[cols].copy()
            for c in ("REPORT_DATE", "NOTICE_DATE", "UPDATE_DATE"):
                if c in out.columns:
                    out[c] = pd.to_datetime(out[c], errors="coerce").dt.normalize()
            for c in ("TOTAL_OPERATE_INCOME", "OPERATE_INCOME", "NETPROFIT", "CONTINUED_NETPROFIT",
                      "DISCONTINUED_NETPROFIT", "PARENT_NETPROFIT", "DEDUCT_PARENT_NETPROFIT"):
                if c in out.columns:
                    out[c] = pd.to_numeric(out[c], errors="coerce")
            out.insert(0, "code", code)
            return out.sort_values("REPORT_DATE").reset_index(drop=True)

        try:
            df = _retry(_fetch, self.retries, what=f"eastmoney income {code}")
        except RuntimeError as exc:
            if isinstance(exc.__cause__, NoData):
                marker.touch()
                raise NoData(code) from exc
            raise
        self.cache.write_df(df, p)
        return df

    # -- dividends / bonus shares -------------------------------------------
    def dividends(self, code: str, refresh: bool = False, max_age_days: float = 7.0) -> pd.DataFrame:
        """分红送配详情: cash dividend per 10 shares (pre-tax), bonus/transfer per 10 shares, ex-date."""
        import akshare as ak

        code = str(code).zfill(6)
        p = self.cache.path("eastmoney", "dividend", f"{code}.parquet")
        marker = p.with_suffix(".nodata")
        if not refresh and marker.exists() and Cache.age_days(marker) < 30:
            return pd.DataFrame(columns=["code", "report_period", "cash_div_per10", "bonus_per10", "ex_date", "status"])
        if p.exists() and not (refresh and Cache.age_days(p) > max_age_days):
            return self.cache.read_df(p)
        time.sleep(self.sleep_s)

        def _fetch():
            try:
                df = ak.stock_fhps_detail_em(symbol=code)
            except (TypeError, KeyError) as exc:  # None payload for delisted companies
                raise NoData(code) from exc
            if df is None or df.empty:
                raise NoData(code)
            out = pd.DataFrame({
                "code": code,
                "report_period": pd.to_datetime(df["报告期"], errors="coerce"),
                "cash_div_per10": pd.to_numeric(df["现金分红-现金分红比例"], errors="coerce"),
                "bonus_per10": pd.to_numeric(df["送转股份-送转总比例"], errors="coerce"),
                "ex_date": pd.to_datetime(df["除权除息日"], errors="coerce"),
                "record_date": pd.to_datetime(df["股权登记日"], errors="coerce"),
                "status": df["方案进度"].astype(str),
            })
            return out.sort_values("ex_date").reset_index(drop=True)

        try:
            df = _retry(_fetch, self.retries, what=f"eastmoney dividends {code}")
        except RuntimeError as exc:
            if isinstance(exc.__cause__, NoData):
                marker.touch()
                return pd.DataFrame(columns=["code", "report_period", "cash_div_per10", "bonus_per10", "ex_date", "record_date", "status"])
            raise
        self.cache.write_df(df, p)
        return df

    # -- share structure ----------------------------------------------------
    def share_structure(self, code: str, refresh: bool = False, max_age_days: float = 7.0) -> pd.DataFrame:
        """股本结构 history: change_date, total_shares, listed_a_shares (流通A股), reason."""
        import akshare as ak

        code = str(code).zfill(6)
        p = self.cache.path("eastmoney", "shares", f"{code}.parquet")
        marker = p.with_suffix(".nodata")
        if not refresh and marker.exists() and Cache.age_days(marker) < 30:
            raise NoData(code)
        if p.exists() and not (refresh and Cache.age_days(p) > max_age_days):
            return self.cache.read_df(p)
        time.sleep(self.sleep_s)

        def _fetch():
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    df = ak.stock_zh_a_gbjg_em(symbol=em_secucode(code))
                except (KeyError, TypeError) as exc:
                    raise NoData(code) from exc
            if df is None or df.empty:
                raise NoData(code)
            out = pd.DataFrame({
                "code": code,
                "change_date": pd.to_datetime(df["变更日期"], errors="coerce"),
                "total_shares": pd.to_numeric(df["总股本"], errors="coerce"),
                "float_shares": pd.to_numeric(df["已流通股份"], errors="coerce"),
                "listed_a_shares": pd.to_numeric(df["已上市流通A股"], errors="coerce"),
                "reason": df["变动原因"].astype(str),
            })
            return out.dropna(subset=["change_date"]).sort_values("change_date").reset_index(drop=True)

        try:
            df = _retry(_fetch, self.retries, what=f"eastmoney shares {code}")
        except RuntimeError as exc:
            if isinstance(exc.__cause__, NoData):
                marker.touch()
                raise NoData(code) from exc
            raise
        self.cache.write_df(df, p)
        return df

    # -- rights issues (market wide) ------------------------------------------
    def rights_issues(self, refresh: bool = False, max_age_days: float = 7.0) -> pd.DataFrame:
        """配股 records for the whole market: code, ratio (per share), price, record_date."""
        import akshare as ak

        p = self.cache.path("eastmoney", "rights_issues.parquet")
        if p.exists() and not (refresh and Cache.age_days(p) > max_age_days):
            return self.cache.read_df(p)
        time.sleep(self.sleep_s)

        def _fetch():
            df = ak.stock_pg_em()
            ratio = df["配股比例"].astype(str).str.extract(r"10配([\d\.]+)")[0]
            out = pd.DataFrame({
                "code": df["股票代码"].astype(str).str.zfill(6),
                "name": df["股票简称"].astype(str),
                "rights_per_share": pd.to_numeric(ratio, errors="coerce") / 10.0,
                "rights_price": pd.to_numeric(df["配股价"], errors="coerce"),
                "record_date": pd.to_datetime(df["股权登记日"], errors="coerce"),
                "listing_date": pd.to_datetime(df["上市日"], errors="coerce") if "上市日" in df.columns else pd.NaT,
            })
            return out.dropna(subset=["record_date", "rights_per_share"]).sort_values("record_date").reset_index(drop=True)

        df = _retry(_fetch, self.retries, what="eastmoney rights issues")
        self.cache.write_df(df, p)
        return df

    # -- performance report (batch, cross-check only) --------------------------
    def yjbb(self, period: str, refresh: bool = False) -> pd.DataFrame:
        import akshare as ak

        p = self.cache.path("eastmoney", "yjbb", f"{period}.parquet")
        if p.exists() and not refresh:
            return self.cache.read_df(p)
        time.sleep(self.sleep_s)
        df = _retry(lambda: ak.stock_yjbb_em(date=period), self.retries, what=f"yjbb {period}")
        self.cache.write_df(df, p)
        return df


# --------------------------------------------------------------------------- #
# CSIndex — official index data & announcements
# --------------------------------------------------------------------------- #
class WafBlocked(Exception):
    pass


class CSIndexClient:
    BASE = "https://www.csindex.com.cn/csindex-home"
    ANN_LIST = BASE + "/announcement/queryAnnouncementByVonew"
    ANN_DETAIL = BASE + "/announcement/queryAnnouncementById"
    CONS_URL = "https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/file/autofile/cons/{code}cons.xls"
    WEIGHT_URL = "https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/file/autofile/closeweight/{code}closeweight.xls"

    def __init__(self, cache: Cache, sleep_s: float = 0.5, detail_sleep_s: float = 3.0, retries: int = 4):
        self.cache = cache
        self.sleep_s = sleep_s
        self.detail_sleep_s = detail_sleep_s
        self.retries = retries

    # -- index daily ---------------------------------------------------------
    def index_daily(self, code: str = "000300", start: str = "20050101", end: Optional[str] = None,
                    refresh: bool = False, max_age_days: float = 1.0) -> pd.DataFrame:
        """Official daily OHLC (+volume/amount when provided) of a CSI index."""
        import akshare as ak

        end = end or datetime.today().strftime("%Y%m%d")
        p = self.cache.path("csindex", "index_daily", f"{code}.parquet")
        if p.exists() and not (refresh and Cache.age_days(p) > max_age_days):
            return self.cache.read_df(p)
        time.sleep(self.sleep_s)

        def _fetch():
            df = ak.stock_zh_index_hist_csindex(symbol=code, start_date=start, end_date=end)
            ren = {"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close",
                   "涨跌": "change", "涨跌幅": "pct_chg", "成交量": "volume", "成交金额": "amount",
                   "样本数量": "n_constituents", "指数代码": "index_code"}
            df = df.rename(columns=ren)
            keep = [c for c in ["date", "index_code", "open", "high", "low", "close", "change", "pct_chg", "volume", "amount", "n_constituents"] if c in df.columns]
            df = df[keep].copy()
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            for c in keep[2:]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            return df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)

        df = _retry(_fetch, self.retries, what=f"csindex daily {code}")
        self.cache.write_df(df, p)
        return df

    # -- current constituents / weights -------------------------------------
    def current_constituents(self, code: str = "000300", refresh: bool = False) -> pd.DataFrame:
        import akshare as ak

        p = self.cache.path("csindex", f"{code}_cons_latest.parquet")
        if p.exists() and not (refresh and Cache.age_days(p) > 1):
            return self.cache.read_df(p)
        time.sleep(self.sleep_s)
        df = _retry(lambda: ak.index_stock_cons_csindex(symbol=code), self.retries, what="csindex cons")
        out = pd.DataFrame({
            "as_of": pd.to_datetime(df["日期"]),
            "code": df["成分券代码"].astype(str).str.zfill(6),
            "name": df["成分券名称"].astype(str),
            "exchange": df["交易所"].astype(str),
        })
        self.cache.write_df(out, p)
        return out

    def current_weights(self, code: str = "000300", refresh: bool = False) -> pd.DataFrame:
        import akshare as ak

        p = self.cache.path("csindex", f"{code}_weights_latest.parquet")
        if p.exists() and not (refresh and Cache.age_days(p) > 1):
            return self.cache.read_df(p)
        time.sleep(self.sleep_s)
        df = _retry(lambda: ak.index_stock_cons_weight_csindex(symbol=code), self.retries, what="csindex weights")
        out = pd.DataFrame({
            "as_of": pd.to_datetime(df["日期"]),
            "code": df["成分券代码"].astype(str).str.zfill(6),
            "name": df["成分券名称"].astype(str),
            "weight": pd.to_numeric(df["权重"], errors="coerce") / 100.0,
        })
        self.cache.write_df(out, p)
        return out

    # -- announcements ---------------------------------------------------------
    def rebalance_announcements(self, refresh: bool = False, max_age_days: float = 1.0,
                                topic: str = "index_rebalance") -> list[dict]:
        """All CSIndex 'index rebalance' announcements (every index), newest first."""
        p = self.cache.path("csindex", "announcements", "rebalance_list.json")
        if p.exists() and not (refresh and Cache.age_days(p) > max_age_days):
            return self.cache.read_json(p)
        payload = {"lang": "cn", "page": {"key": "", "page": 1, "rows": 200, "desc": "", "sortBy": ""},
                   "classList": ["index"], "indexList": [], "relatedTopics": [topic], "typeList": ["announcement"]}
        items: list[dict] = []
        page = 1
        while True:
            payload["page"]["page"] = page
            time.sleep(self.sleep_s)
            j = _retry(lambda: requests.post(self.ANN_LIST, json=payload, timeout=60).json(), self.retries,
                       what=f"csindex announcements p{page}")
            data = j.get("data") or []
            items.extend(data)
            total_pages = int(j.get("size") or 1)
            if page >= total_pages or not data:
                break
            page += 1
        self.cache.write_json(items, p)
        return items

    def announcement_detail(self, ann_id: int, refresh: bool = False) -> dict:
        """Announcement body + enclosure list. The endpoint sits behind a WAF: pace requests."""
        p = self.cache.path("csindex", "announcements", "detail", f"{ann_id}.json")
        if p.exists() and not refresh:
            return self.cache.read_json(p)
        blocked_waits = [60, 120, 300, 600]
        for attempt in range(len(blocked_waits) + 1):
            time.sleep(self.detail_sleep_s)
            r = requests.get(self.ANN_DETAIL, params={"id": ann_id}, timeout=60)
            if r.status_code == 200 and r.text.lstrip().startswith("{"):
                data = r.json().get("data") or {}
                self.cache.write_json(data, p)
                return data
            if attempt < len(blocked_waits):
                wait = blocked_waits[attempt]
                log.warning("CSIndex detail %s blocked (HTTP %s); waiting %ss", ann_id, r.status_code, wait)
                time.sleep(wait)
        raise WafBlocked(f"CSIndex announcement detail {ann_id} unavailable (WAF)")

    def download(self, url: str, refresh: bool = False) -> Path:
        """Download an enclosure (xls/xlsx/pdf/doc) to data/raw/csindex/files, return the path."""
        name = re.sub(r"[^\w\.\-\u4e00-\u9fff]+", "_", url.split("/")[-1])[:150]
        p = self.cache.path("csindex", "files", name)
        if p.exists() and p.stat().st_size > 0 and not refresh:
            return p
        time.sleep(self.sleep_s)

        def _fetch():
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            if len(r.content) < 200:
                raise RuntimeError(f"suspiciously small download {url}")
            return r.content

        content = _retry(_fetch, self.retries, what=f"download {url}")
        with open(p, "wb") as fh:
            fh.write(content)
        return p


# --------------------------------------------------------------------------- #
# Facade
# --------------------------------------------------------------------------- #
class DataHub:
    """Wires the clients to the configuration; the rest of the package only talks to this."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        dcfg = cfg["data"]
        self.raw = Cache(abs_path(cfg, dcfg["raw_dir"]))
        self.processed = abs_path(cfg, dcfg["processed_dir"])
        self.processed.mkdir(parents=True, exist_ok=True)
        self.sina = SinaClient(self.raw, sleep_s=max(dcfg.get("request_sleep_s", 0.35), 0.5), retries=dcfg.get("max_retries", 4))
        self.em = EastmoneyClient(self.raw, sleep_s=dcfg.get("request_sleep_s", 0.35), retries=dcfg.get("max_retries", 4))
        self.csi = CSIndexClient(self.raw, detail_sleep_s=dcfg.get("csindex_detail_sleep_s", 3.0), retries=dcfg.get("max_retries", 4))
        self.provider = dcfg.get("provider", "akshare")
        if self.provider == "tushare":
            from .tushare_provider import TushareProvider  # optional

            self.tushare = TushareProvider(self.raw, cfg)

    # convenience passthroughs -------------------------------------------------
    def index_daily(self, code: Optional[str] = None, refresh: bool = False) -> pd.DataFrame:
        code = code or self.cfg["data"]["parent_index_code"]
        return self.csi.index_daily(code, start=self.cfg["data"]["start_date"].replace("-", ""), refresh=refresh)

    def stock_daily(self, code: str, refresh: bool = False) -> pd.DataFrame:
        if self.provider == "tushare":
            return self.tushare.stock_daily(code, refresh=refresh)
        return self.sina.daily(code, refresh=refresh)

    def income(self, code: str, refresh: bool = False) -> pd.DataFrame:
        if self.provider == "tushare":
            return self.tushare.income(code, refresh=refresh)
        return self.em.income_statement(code, refresh=refresh)

    def dividends(self, code: str, refresh: bool = False) -> pd.DataFrame:
        if self.provider == "tushare":
            return self.tushare.dividends(code, refresh=refresh)
        return self.em.dividends(code, refresh=refresh)

    def shares(self, code: str, refresh: bool = False) -> pd.DataFrame:
        if self.provider == "tushare":
            return self.tushare.shares(code, refresh=refresh)
        return self.em.share_structure(code, refresh=refresh)

    def rights_issues(self, refresh: bool = False) -> pd.DataFrame:
        return self.em.rights_issues(refresh=refresh)

    def holders(self, code: str, refresh: bool = False) -> pd.DataFrame:
        return self.sina.shareholders(code, refresh=refresh)
