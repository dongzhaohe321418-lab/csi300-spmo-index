"""Point-in-time CSI 300 membership rebuilt from official CSIndex announcements.

Approach
--------
1. Pull every CSIndex "index rebalance" announcement whose title names 沪深300 and
   that actually changes the CSI 300 constituent list (regular semi-annual reviews,
   temporary adjustments, supplements). Sub-index announcements (行业/风格/精明…) are
   excluded.
2. Parse each announcement's add/remove list for the CSI 300 (HTML tables, linked or
   attached xls/xlsx, PDF attachments) and its effective date. Effective dates that
   depend on a delisting ("自退市之日起") or a listing ("自…上市之日起") are resolved
   from the securities' actual trading history.
3. Start from the current official constituent list and walk backwards through the
   events, undoing each adjustment.  Every intermediate list must contain exactly
   300 securities and the list reconstructed for 2005-07-01 must equal the official
   full list published by CSIndex (anchor).  Any mismatch aborts the run.
4. Emit membership intervals [start_date, end_date) per security; `members_at(date)`
   answers the point-in-time question for any trading day since 2005-04-08.

Nothing here uses today's list to infer history other than as the starting point of
the backward reconstruction, which is validated against the 2005 anchor.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from io import StringIO
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from .data import DataHub, NoData
from .utils import TradingCalendar

log = logging.getLogger(__name__)

CODE_RE = re.compile(r"(?<!\d)([036]\d{5})(?!\d)")
TITLE_RE = re.compile(r"沪深\s*300")
EXCLUDE_TITLE_RE = re.compile(
    r"精明|行业|风格|主题|价值|成长|红利|波动|等权|非周期|周期|相对|ESG|备选|指数系列|新指数|发布|"
    r"地产|金融|医药|能源|消费|工业|材料|电信|公用|信息|可选|股份类别"
)
FULL_LIST_RE = re.compile(r"沪深300指数样本股名单")
OUT_KW = re.compile(r"调出|剔除|删除|移出")
IN_KW = re.compile(r"调入|新进|纳入|新增")
INDEX_COL_KW = re.compile(r"指数(名称|简称|代码)")
CSI300_ROW_RE = re.compile(r"^(000300|沪深300(指数)?)$")

# Manual corrections for announcements whose machine-readable content is incomplete.
# Each entry is documented in README (data section).
MANUAL_FIXES: dict[int, dict] = {
    # 2013-08-13: 美的电器(000527) absorbed by 美的集团, whose code was not yet assigned
    # in the table ("-"); 美的集团 listed as 000333 on 2013-09-18.
    3453: {"adds": {"000333": "美的集团"}, "removes": {"000527": "美的电器"}, "effective_rule": "listing:000333"},
    # 2008-06-04 #907 is a verbatim re-publication of 2008-03-10 #38 (东方锅炉) -> skip.
    907: {"skip": True},
    # 2005-07-12 #86 is an incomplete copy (297 codes) of the 2005-07-22 full list (#6773).
    86: {"skip": True},
    # 2006-04-14 #92 (prose only): 齐鲁石化/石油大明/扬子石化/中原油气 taken private by
    # tender offer and delisted; replaced in order by G燃气/航天信息/G张裕/上电股份.
    # All four last traded on 2006-04-06, so the replacements are effective together.
    92: {"adds": {"000793": "G燃气", "600271": "航天信息", "000869": "G张裕", "600627": "上电股份"},
         "removes": {"600002": "齐鲁石化", "000406": "石油大明", "000866": "扬子石化", "000956": "中原油气"}},
    # The main announcement of the July-2008 regular review is mis-filed in the CSIndex
    # archive (#907 repeats the 东方锅炉 notice). The companion document
    # "沪深300行业指数样本股调样名单" (#14, same day) lists every add/remove by sector;
    # the union of the sector lists (net of sector reclassifications) is the CSI 300 change
    # list. Validated against 2008-12 where both documents exist (identical result).
    14: {"parser": "sector_union", "kind": "regular", "effective_date": "2008-07-01"},
}

# Announcements whose title does not name 沪深300 but that may still carry a 沪深300 row
# (delistings / mergers / spin-offs). They are used only when a 沪深300 row is found.
CORP_EVENT_TITLE_RE = re.compile(r"退市|退巿|终止上市|吸收合并|分立|临时调整指数样本")
NOISE_TITLE_RE = re.compile(r"三板|债|海外|港|新股|精明|中华|指数系列|样本券|深证|国证|基金|ETF|商品|期货|美元|全球|亚洲|台湾|日本|欧洲|美国|B股")


@dataclass
class AdjustmentEvent:
    ann_id: int
    publish_date: str
    title: str
    kind: str                       # regular | temporary | full_list | supplement
    effective_date: Optional[str]   # YYYY-MM-DD (first trading day the new list is in force)
    adds: dict = field(default_factory=dict)      # code -> name
    removes: dict = field(default_factory=dict)   # code -> name
    expected_n: Optional[int] = None
    source: str = ""
    note: str = ""
    pending: bool = False


# --------------------------------------------------------------------------- #
# text helpers
# --------------------------------------------------------------------------- #
def html_to_text(html: str) -> str:
    txt = re.sub(r"<br\s*/?>|</p>|</tr>|</div>", "\n", html or "", flags=re.I)
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = txt.replace("&nbsp;", " ").replace("&amp;", "&")
    txt = re.sub(r"[ \t\u3000]+", " ", txt)
    txt = re.sub(r"\n\s*\n+", "\n", txt)
    return txt.strip()


def _clean_name(name: str) -> str:
    name = re.sub(r"[（(].*?[)）]", "", str(name))
    name = re.sub(r"\s+", "", name)
    return "" if name in ("nan", "None", "-", "") else name


def expected_count_from_text(text: str) -> Optional[int]:
    t = re.sub(r"\s+", "", text)
    m = re.search(r"沪深300指数(样本股?)?(更换|调整|调换)(\d+)只", t)
    return int(m.group(3)) if m else None


def classify(title: str, text: str) -> str:
    if FULL_LIST_RE.search(title):
        return "full_list"
    if "补充公告" in title:
        return "supplement"
    t = re.sub(r"\s+", "", text[:400])
    if re.search(r"定期调整结果", title) or re.search(r"经指数专家委员会审议", t):
        return "regular"
    return "temporary"


# --------------------------------------------------------------------------- #
# generic change-table parser
# --------------------------------------------------------------------------- #
def _norm_cell(x) -> str:
    s = "" if x is None else str(x)
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", "", s)
    return "" if s.lower() in ("nan", "none") else s


def parse_change_table(df: pd.DataFrame) -> Optional[dict]:
    """Parse one 调出/调入 table.

    Returns {"adds", "removes", "multi_index", "mentions_csi300"} or None if the table
    is not a change table.  For multi-index tables only the 沪深300 rows are used.
    """
    vals = [[_norm_cell(c) for c in row] for row in df.values.tolist()]
    if not vals:
        return None
    ncol = max(len(r) for r in vals)
    vals = [r + [""] * (ncol - len(r)) for r in vals]
    # wrapper tables (huge cells) are not change tables
    if any(len(c) > 60 for r in vals[:3] for c in r):
        return None
    side = [None] * ncol
    header_rows = 0
    multi_index = False
    for i, row in enumerate(vals[:4]):
        has_kw = False
        for j, c in enumerate(row):
            if OUT_KW.search(c) and not CODE_RE.search(c):
                side[j] = "out"
                has_kw = True
            elif IN_KW.search(c) and not CODE_RE.search(c):
                side[j] = "in"
                has_kw = True
            if INDEX_COL_KW.search(c):
                multi_index = True
        if has_kw or any(k in "".join(row) for k in ("股票代码", "证券代码", "指数名称", "指数简称", "指数代码", "调整名单")):
            header_rows = i + 1
    if all(s is None for s in side):
        return None
    # propagate side labels to neighbouring unlabeled columns (name columns)
    for j in range(ncol):
        if side[j] is None and j > 0 and side[j - 1] is not None and not multi_index:
            side[j] = side[j - 1]
    for j in range(ncol - 1, -1, -1):
        if side[j] is None and j + 1 < ncol and side[j + 1] is not None and j > (1 if multi_index else -1):
            side[j] = side[j + 1]
    if multi_index:
        # first labelled column: everything before it is index identification
        first = next(j for j in range(ncol) if side[j] is not None)
        # fill gaps inside labelled area
        cur = None
        for j in range(first, ncol):
            if side[j] is None:
                side[j] = cur
            else:
                cur = side[j]
    body = vals[header_rows:]
    first_data_col = next(j for j in range(ncol) if side[j] is not None)
    if not multi_index:
        # heuristic: rows whose first cell is an index name -> multi-index table
        idx_like = sum(1 for r in body if r and re.search(r"指数|沪深|中证|上证|300|500|800", r[0]) and not CODE_RE.fullmatch(r[0]))
        if idx_like >= max(2, len(body) // 2):
            multi_index = True
    mentions = any("沪深300" in c for r in vals[:2] for c in r)
    adds, removes = {}, {}
    for row in body:
        if multi_index:
            key = [c for c in row[:first_data_col] if c]
            if not any(CSI300_ROW_RE.match(k) for k in key):
                continue
        for j, c in enumerate(row):
            if side[j] is None or not CODE_RE.fullmatch(c):
                continue
            # name: adjacent non-code cell on the same side
            name = ""
            for jj in (j + 1, j - 1):
                if 0 <= jj < ncol and side[jj] == side[j] and row[jj] and not CODE_RE.fullmatch(row[jj]):
                    name = _clean_name(row[jj])
                    break
            (removes if side[j] == "out" else adds)[c] = name
    return {"adds": adds, "removes": removes, "multi_index": multi_index, "mentions_csi300": mentions}


def _inner_tables(html: str) -> list[pd.DataFrame]:
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("beautifulsoup4 required") from exc
    soup = BeautifulSoup(html or "", "lxml")
    out = []
    for tb in soup.find_all("table"):
        if tb.find("table"):
            continue
        try:
            out.append(pd.read_html(StringIO(str(tb)), header=None)[0])
        except ValueError:
            continue
    return out


def parse_html_changes(content: str) -> tuple[dict, dict, str]:
    tables = _inner_tables(content)
    parsed = [(tb, parse_change_table(tb)) for tb in tables]
    parsed = [(tb, p) for tb, p in parsed if p is not None]
    if not parsed:
        return {}, {}, "html:none"
    # 1) multi-index tables that contain a 沪深300 row (an announcement may hold several,
    #    e.g. one table per delisting event)
    adds, removes = {}, {}
    for tb, p in parsed:
        if p["multi_index"] and (p["adds"] or p["removes"]):
            adds.update(p["adds"])
            removes.update(p["removes"])
    if adds or removes:
        return adds, removes, "html:multi"
    # 2) paired tables whose caption mentions 沪深300
    for tb, p in parsed:
        if not p["multi_index"] and p["mentions_csi300"] and (p["adds"] or p["removes"]):
            return p["adds"], p["removes"], "html:paired"
    # 3) first paired table (announcement titles list 沪深300 first)
    for tb, p in parsed:
        if not p["multi_index"] and (p["adds"] or p["removes"]):
            return p["adds"], p["removes"], "html:first"
    return {}, {}, "html:none"


def parse_excel_changes(path: Path) -> tuple[dict, dict, str]:
    xl = pd.ExcelFile(path)
    names = xl.sheet_names
    adds, removes = {}, {}
    if any("调入" in s for s in names) and any("调出" in s for s in names):
        # format A: sheets 调入 / 调出 with 指数代码, 指数简称, 证券代码, 证券简称
        for sheet, target in (("调入", adds), ("调出", removes)):
            sh = next(s for s in names if sheet in s)
            df = xl.parse(sh, header=None, dtype=str)
            for row in df.itertuples(index=False):
                cells = [_norm_cell(c) for c in row]
                if len(cells) < 3:
                    continue
                if cells[0].zfill(6) == "000300" or cells[1] == "沪深300":
                    for j, c in enumerate(cells[2:], start=2):
                        if CODE_RE.fullmatch(c.zfill(6)) and len(c) <= 6:
                            target[c.zfill(6)] = _clean_name(cells[j + 1]) if j + 1 < len(cells) else ""
                            break
        return adds, removes, "excel:A"
    # format B: one sheet, one row per index with 调出 pairs then 调入 pairs
    for sh in names:
        df = xl.parse(sh, header=None, dtype=str)
        p = parse_change_table(df)
        if p and (p["adds"] or p["removes"]):
            return p["adds"], p["removes"], "excel:B"
    return adds, removes, "excel:none"


def parse_pdf_changes(path: Path) -> tuple[dict, dict, str]:
    import pdfplumber

    lines: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            lines.extend(txt.splitlines())
    adds, removes = {}, {}
    in_section = False
    for ln in lines:
        s = ln.strip()
        if re.search(r"沪深\s*300\s*指数样本调整名单", s):
            in_section = True
            continue
        if in_section and re.search(r"指数样本调整名单", s):
            break
        if not in_section:
            continue
        toks = s.split()
        codes = [(i, t) for i, t in enumerate(toks) if CODE_RE.fullmatch(t)]
        if not codes:
            continue
        if len(codes) == 2:
            (i1, c1), (i2, c2) = codes
            removes[c1] = _clean_name(toks[i1 + 1]) if i1 + 1 < len(toks) and not CODE_RE.fullmatch(toks[i1 + 1]) else ""
            adds[c2] = _clean_name(toks[i2 + 1]) if i2 + 1 < len(toks) else ""
        elif len(codes) == 1:
            i1, c1 = codes[0]
            name = _clean_name(toks[i1 + 1]) if i1 + 1 < len(toks) else ""
            (removes if i1 == 0 and len(toks) <= 2 else adds)[c1] = name
    return adds, removes, "pdf"


def parse_full_list(text: str, content: str, files: list[Path]) -> dict:
    """The 2005-07-01 full constituent list (anchor)."""
    codes: dict = {}
    for f in files:
        if f.suffix.lower() in (".xls", ".xlsx"):
            xl = pd.ExcelFile(f)
            for sh in xl.sheet_names:
                df = xl.parse(sh, header=None, dtype=str).fillna("")
                for row in df.itertuples(index=False):
                    cells = [_norm_cell(c) for c in row]
                    for j, c in enumerate(cells):
                        if CODE_RE.fullmatch(c.zfill(6)) and len(c) <= 6:
                            codes[c.zfill(6)] = _clean_name(cells[j + 1]) if j + 1 < len(cells) else ""
    if not codes:
        for tb in _inner_tables(content):
            for row in tb.values.tolist():
                cells = [_norm_cell(c) for c in row]
                for j, c in enumerate(cells):
                    if CODE_RE.fullmatch(c):
                        codes[c] = _clean_name(cells[j + 1]) if j + 1 < len(cells) else ""
    if not codes:
        for m in re.finditer(r"([036]\d{5})\s+([\u4e00-\u9fffA-Za-z\*\d]{2,10})", text):
            codes[m.group(1)] = m.group(2)
    return codes


# --------------------------------------------------------------------------- #
# Effective-date resolution
# --------------------------------------------------------------------------- #
def _cn_date(y, m, d) -> pd.Timestamp:
    return pd.Timestamp(year=int(y), month=int(m), day=int(d))


def explicit_effective_date(text: str, publish_date: str, cal: TradingCalendar) -> Optional[pd.Timestamp]:
    """Effective date stated explicitly in the announcement text (None if data-dependent)."""
    t = re.sub(r"\s+", "", text)
    pub = pd.Timestamp(publish_date)
    # "...于2021年6月11日收盘后生效" / "收市后生效" -> next trading day
    m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日收[市盘]后", t)
    if m:
        return cal.next(cal.last_on_or_before(_cn_date(*m.groups())), 1)
    # "于2008年1月第一个交易日调整"
    m = re.search(r"(\d{4})年(\d{1,2})月第一个交易日", t)
    if m:
        return cal.first_on_or_after(_cn_date(m.group(1), m.group(2), 1))
    # "（2005年7月1日起生效）" / "自2015年12月30日起" / "于2010年1月4日一并实施" / "于2010年7月1日调整"
    for pat in (
        r"(\d{4})年(\d{1,2})月(\d{1,2})日起(正式)?(生效|实施|调整)?",
        r"自(\d{4})年(\d{1,2})月(\d{1,2})日",
        r"于(\d{4})年(\d{1,2})月(\d{1,2})日",
        r"(\d{4})年(\d{1,2})月(\d{1,2})日(正式)?(生效|实施|调整)",
    ):
        m = re.search(pat, t)
        if m:
            return cal.first_on_or_after(_cn_date(m.group(1), m.group(2), m.group(3)))
    # "决定于7月3日调整" / "自7月19日起" (year omitted -> publication year, or next year if it wrapped)
    m = re.search(r"[于自](\d{1,2})月(\d{1,2})日", t)
    if m:
        y = pub.year
        d = _cn_date(y, m.group(1), m.group(2))
        if d < pub - pd.Timedelta(days=30):
            d = _cn_date(y + 1, m.group(1), m.group(2))
        return cal.first_on_or_after(d)
    m = re.search(r"于(\d{1,2})月第一个交易日", t)
    if m:
        y = pub.year if int(m.group(1)) >= pub.month else pub.year + 1
        return cal.first_on_or_after(_cn_date(y, m.group(1), 1))
    return None


def data_driven_effective_dates(ev: AdjustmentEvent, text: str, hub: DataHub, cal: TradingCalendar,
                                data_end: pd.Timestamp) -> list[AdjustmentEvent]:
    """Resolve '自…退市之日起' / '自…上市之日起' style dates from actual trading history.

    Returns one or more events (one per distinct effective date); events whose
    trigger has not happened yet are flagged pending.
    """
    t = re.sub(r"\s+", "", text)
    pub_next = cal.next(cal.last_on_or_before(pd.Timestamp(ev.publish_date)), 1)
    rule = MANUAL_FIXES.get(ev.ann_id, {}).get("effective_rule")
    if rule and rule.startswith("listing:"):
        code = rule.split(":")[1]
        try:
            first = hub.stock_daily(code)["date"].min()
        except NoData:
            first = None
        if first is None or first > data_end:
            ev.pending = True
            ev.note = f"waiting for listing of {code}"
            return [ev]
        ev.effective_date = cal.first_on_or_after(first).strftime("%Y-%m-%d")
        ev.note = f"effective on listing day of {code}"
        return [ev]
    if re.search(r"退市|终止上市", t):
        # removal on the day the security ceases to be listed: the first trading day
        # after its last traded day, but never before the announcement is public.
        groups: dict[pd.Timestamp, tuple[dict, dict]] = {}
        adds_list = list(ev.adds.items())
        rem_list = list(ev.removes.items())
        for i, (code, name) in enumerate(rem_list):
            try:
                last = hub.stock_daily(code)["date"].max()
            except NoData:
                last = None
            if last is None or last >= cal.prev(data_end, 3):
                # still trading -> delisting has not happened yet
                ev.pending = True
                ev.note = f"{code} still trading; delisting pending"
                return [ev]
            eff = max(cal.next(last, 1), pub_next)
            a = dict([adds_list[i]]) if len(adds_list) == len(rem_list) else (dict(adds_list) if i == 0 else {})
            g = groups.setdefault(eff, ({}, {}))
            g[0].update(a)
            g[1][code] = name
        out = []
        for eff, (a, r) in sorted(groups.items()):
            e2 = AdjustmentEvent(ev.ann_id, ev.publish_date, ev.title, ev.kind, eff.strftime("%Y-%m-%d"),
                                 a, r, ev.expected_n, ev.source, note="effective on delisting of removed security")
            out.append(e2)
        return out or [ev]
    return [ev]


# --------------------------------------------------------------------------- #
# Event construction
# --------------------------------------------------------------------------- #
def relevant_announcements(all_anns: list[dict]) -> list[dict]:
    """Announcements to parse. Each gets a `_role`: 'primary' (title names 沪深300),
    'candidate' (corporate event, used only if a 沪深300 row is found) or 'companion'
    (explicitly configured in MANUAL_FIXES)."""
    out = []
    for a in all_anns:
        title = a.get("title", "")
        fix = MANUAL_FIXES.get(a["id"], {})
        if fix.get("parser"):
            out.append({**a, "_role": "companion"})
        elif TITLE_RE.search(title) and not EXCLUDE_TITLE_RE.search(title):
            out.append({**a, "_role": "primary"})
        elif CORP_EVENT_TITLE_RE.search(title) and not NOISE_TITLE_RE.search(title):
            out.append({**a, "_role": "candidate"})
    out.sort(key=lambda a: (a["publishDate"], a["id"]))
    return out


def parse_sector_union(content: str) -> tuple[dict, dict, str]:
    """Union of the per-sector 调出/调入 tables of a 沪深300行业指数 companion document.
    A security appearing on both sides merely changed sector and is dropped."""
    adds, removes = {}, {}
    for tb in _inner_tables(content):
        p = parse_change_table(tb)
        if p and not p["multi_index"]:
            adds.update(p["adds"])
            removes.update(p["removes"])
    both = set(adds) & set(removes)
    return ({k: v for k, v in adds.items() if k not in both},
            {k: v for k, v in removes.items() if k not in both}, "sector_union")


def _enclosure_files(hub: DataHub, detail: dict) -> list[Path]:
    files = []
    urls = []
    for e in detail.get("enclosureList") or []:
        url = e.get("fileUrl")
        if url and re.search(r"\.(xlsx?|pdf|docx?)$", url, re.I):
            urls.append(url)
    urls += re.findall(r'href="(https?://[^"]+\.(?:xlsx?|pdf))"', detail.get("content") or "", flags=re.I)
    for url in urls:
        try:
            files.append(hub.csi.download(url))
        except Exception as exc:  # noqa: BLE001
            log.warning("could not download %s: %s", url, exc)
    return files


def build_events(hub: DataHub, cal: TradingCalendar, data_end: Optional[pd.Timestamp] = None,
                 refresh: bool = False) -> list[AdjustmentEvent]:
    data_end = pd.Timestamp(data_end) if data_end is not None else cal.days[-1]
    anns = relevant_announcements(hub.csi.rebalance_announcements(refresh=refresh))
    events: list[AdjustmentEvent] = []
    for a in anns:
        fix = MANUAL_FIXES.get(a["id"], {})
        if fix.get("skip"):
            continue
        role = a.get("_role", "primary")
        detail = hub.csi.announcement_detail(a["id"])
        content = detail.get("content") or ""
        text = html_to_text(content)
        kind = fix.get("kind") or classify(a["title"], text)
        if fix.get("effective_date"):
            eff = cal.first_on_or_after(pd.Timestamp(fix["effective_date"]))
        else:
            eff = explicit_effective_date(a["title"] + "\n" + text, a["publishDate"], cal)
        ev = AdjustmentEvent(ann_id=a["id"], publish_date=a["publishDate"], title=a["title"], kind=kind,
                             effective_date=eff.strftime("%Y-%m-%d") if eff is not None else None,
                             expected_n=expected_count_from_text(text))
        files = _enclosure_files(hub, detail)
        if kind == "full_list":
            ev.adds = parse_full_list(text, content, files)
            ev.source = "full_list"
            events.append(ev)
            continue
        adds, removes, src = {}, {}, ""
        if fix.get("parser") == "sector_union":
            adds, removes, src = parse_sector_union(content)
        else:
            for f in files:
                try:
                    if f.suffix.lower() in (".xls", ".xlsx"):
                        adds, removes, src = parse_excel_changes(f)
                    elif f.suffix.lower() == ".pdf":
                        adds, removes, src = parse_pdf_changes(f)
                except Exception as exc:  # noqa: BLE001
                    log.warning("parse failure %s: %s", f, exc)
                if adds or removes:
                    break
            if not (adds or removes):
                adds, removes, src = parse_html_changes(content)
                if role == "candidate" and src != "html:multi":
                    adds, removes = {}, {}      # no explicit 沪深300 row -> not a CSI 300 event
        if role == "candidate" and not (adds or removes):
            continue
        if "adds" in fix:
            adds, src = dict(fix["adds"]), src + "+manual"
        if "removes" in fix:
            removes = dict(fix["removes"])
        ev.adds, ev.removes, ev.source = adds, removes, src
        if role == "candidate":
            ev.note = "title does not name 沪深300; 沪深300 row found in body"
        if ev.expected_n is not None and (len(adds) != ev.expected_n or len(removes) != ev.expected_n):
            log.warning("#%s %s: parsed +%d/-%d but announcement says %d", ev.ann_id, ev.publish_date, len(adds), len(removes), ev.expected_n)
        if ev.effective_date is None or MANUAL_FIXES.get(ev.ann_id, {}).get("effective_rule"):
            events.extend(data_driven_effective_dates(ev, text, hub, cal, data_end))
        else:
            events.append(ev)
    # drop exact duplicates (same effective date & same changes)
    seen = set()
    uniq = []
    for e in events:
        key = (e.effective_date, tuple(sorted(e.adds)), tuple(sorted(e.removes)))
        if e.kind != "full_list" and key in seen:
            log.info("dropping duplicate announcement #%s (%s)", e.ann_id, e.publish_date)
            continue
        seen.add(key)
        uniq.append(e)
    return uniq


def events_to_frame(events: Iterable[AdjustmentEvent]) -> pd.DataFrame:
    rows = []
    for ev in events:
        if ev.kind == "full_list":
            continue
        for code, name in ev.adds.items():
            rows.append((ev.ann_id, ev.publish_date, ev.effective_date, ev.kind, "add", code, name, ev.pending))
        for code, name in ev.removes.items():
            rows.append((ev.ann_id, ev.publish_date, ev.effective_date, ev.kind, "remove", code, name, ev.pending))
    return pd.DataFrame(rows, columns=["ann_id", "publish_date", "effective_date", "kind", "action", "code", "name", "pending"])


# --------------------------------------------------------------------------- #
# Reconstruction
# --------------------------------------------------------------------------- #
class UniverseError(RuntimeError):
    pass


def reconstruct_membership(events: list[AdjustmentEvent], current: pd.DataFrame, expected_size: int = 300,
                           inception: str = "2005-04-08") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk backwards from the current official list; return (intervals, audit).

    intervals: code, name, start_date, end_date (exclusive; NaT = still a member)
    audit:     one row per event with the member count in force before/after it
    """
    cur_asof = pd.Timestamp(current["as_of"].iloc[0]).normalize()
    names = dict(zip(current["code"], current["name"]))
    change_events = [e for e in events if e.kind != "full_list" and e.effective_date and not e.pending and (e.adds or e.removes)]
    for e in events:
        for d in (e.adds, e.removes):
            for c, n in d.items():
                if n:
                    names.setdefault(c, n)
    latest = set(current["code"])
    fwd = sorted([e for e in change_events if pd.Timestamp(e.effective_date) > cur_asof], key=lambda e: e.effective_date)
    for e in fwd:
        latest = (latest - set(e.removes)) | set(e.adds)
    past = sorted([e for e in change_events if pd.Timestamp(e.effective_date) <= cur_asof],
                  key=lambda e: (e.effective_date, e.publish_date), reverse=True)
    problems: list[str] = []
    audit_rows = []
    s = set(latest)
    for e in past:
        missing_adds = set(e.adds) - s
        present_removes = set(e.removes) & s
        if missing_adds:
            problems.append(f"{e.publish_date} #{e.ann_id}: added securities absent from the later list {sorted(missing_adds)}")
        if present_removes:
            problems.append(f"{e.publish_date} #{e.ann_id}: removed securities still present in the later list {sorted(present_removes)}")
        n_after = len(s)
        s = (s - set(e.adds)) | set(e.removes)
        audit_rows.append((e.effective_date, e.publish_date, e.ann_id, e.kind, len(e.adds), len(e.removes), len(s), n_after))
        if len(s) != expected_size:
            problems.append(f"{e.publish_date} #{e.ann_id}: {len(s)} members in force before this event (expected {expected_size})")
    earliest = set(s)
    # anchor check against the official full list
    full = [e for e in events if e.kind == "full_list" and len(e.adds) == expected_size]
    if full:
        anchor = full[-1]
        anchor_set = set(anchor.adds)
        anchor_date = pd.Timestamp(anchor.effective_date or "2005-07-01")
        s_run = set(earliest)
        for e in sorted(past, key=lambda e: (e.effective_date, e.publish_date)):
            if pd.Timestamp(e.effective_date) <= anchor_date:
                s_run = (s_run - set(e.removes)) | set(e.adds)
        diff1 = sorted(s_run - anchor_set)
        diff2 = sorted(anchor_set - s_run)
        if diff1 or diff2:
            problems.append(f"anchor mismatch on {anchor_date.date()}: reconstructed-not-in-anchor={diff1} anchor-not-in-reconstructed={diff2}")
        for c, n in anchor.adds.items():
            if n:
                names.setdefault(c, n)
    else:
        problems.append("no full-list anchor announcement found")
    if problems:
        for p in problems:
            log.error("UNIVERSE: %s", p)
        raise UniverseError("point-in-time CSI300 reconstruction failed:\n" + "\n".join(problems))
    # intervals, chronologically
    chrono = sorted(change_events, key=lambda e: (e.effective_date, e.publish_date))
    open_int: dict[str, pd.Timestamp] = {c: pd.Timestamp(inception) for c in earliest}
    rows = []
    for e in chrono:
        eff = pd.Timestamp(e.effective_date)
        for c in e.removes:
            if c in open_int:
                rows.append((c, names.get(c, ""), open_int.pop(c), eff))
            else:
                log.warning("remove of non-member %s at %s (#%s)", c, eff.date(), e.ann_id)
        for c in e.adds:
            if c in open_int:
                log.warning("add of existing member %s at %s (#%s)", c, eff.date(), e.ann_id)
            else:
                open_int[c] = eff
    for c, st in open_int.items():
        rows.append((c, names.get(c, ""), st, pd.NaT))
    intervals = (pd.DataFrame(rows, columns=["code", "name", "start_date", "end_date"])
                 .sort_values(["code", "start_date"]).reset_index(drop=True))
    audit = pd.DataFrame(audit_rows, columns=["effective_date", "publish_date", "ann_id", "kind", "n_add", "n_remove",
                                              "n_before", "n_after"]).sort_values("effective_date").reset_index(drop=True)
    return intervals, audit


def members_at(intervals: pd.DataFrame, d) -> set:
    d = pd.Timestamp(d).normalize()
    m = (intervals["start_date"] <= d) & (intervals["end_date"].isna() | (intervals["end_date"] > d))
    return set(intervals.loc[m, "code"])


def save_events(events: list[AdjustmentEvent], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([asdict(e) for e in events], fh, ensure_ascii=False, indent=1)


def load_events(path: Path) -> list[AdjustmentEvent]:
    with open(path, "r", encoding="utf-8") as fh:
        return [AdjustmentEvent(**d) for d in json.load(fh)]


def build_universe(hub: DataHub, cal: TradingCalendar, refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, list[AdjustmentEvent]]:
    """Full pipeline: announcements -> events -> validated membership intervals (saved to data/processed)."""
    events = build_events(hub, cal, refresh=refresh)
    current = hub.csi.current_constituents(refresh=refresh)
    intervals, audit = reconstruct_membership(events, current, expected_size=hub.cfg["universe"]["expected_size"])
    out = hub.processed
    save_events(events, out / "csi300_adjustment_events.json")
    events_to_frame(events).to_csv(out / "csi300_adjustment_events.csv", index=False, encoding="utf-8-sig")
    intervals.to_csv(out / "csi300_membership_intervals.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(out / "csi300_reconstruction_audit.csv", index=False, encoding="utf-8-sig")
    return intervals, audit, events
