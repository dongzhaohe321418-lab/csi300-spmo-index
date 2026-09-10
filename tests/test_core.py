"""Unit tests for the fixed strategy logic (no network access required)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import financials as F  # noqa: E402
from src import metrics as M  # noqa: E402
from src import plots as P  # noqa: E402
from src import selection as S  # noqa: E402
from src import universe as U  # noqa: E402
from src import weighting as W  # noqa: E402
from src.index_engine import IndexEngine, MembershipError, Rebalance, cost_rates  # noqa: E402
from src.momentum import compute_momentum, momentum_window  # noqa: E402
from src.utils import TradingCalendar, months_before  # noqa: E402


# --------------------------------------------------------------------------- calendar
def _cal(start="2004-01-01", end="2026-12-31"):
    days = pd.bdate_range(start, end)
    return TradingCalendar(days)


def test_calendar_month_end_and_third_friday():
    cal = _cal()
    assert cal.month_end(2026, 8) == pd.Timestamp("2026-08-31")
    assert cal.month_end(2025, 8) == pd.Timestamp("2025-08-29")      # Aug-30/31 2025 is a weekend
    assert cal.third_friday(2026, 9) == pd.Timestamp("2026-09-18")
    assert cal.third_friday(2005, 9) == pd.Timestamp("2005-09-16")
    assert months_before("2026-08-31", 1) == pd.Timestamp("2026-07-31")
    assert months_before("2026-08-31", 13) == pd.Timestamp("2025-07-31")
    assert months_before("2024-03-31", 1) == pd.Timestamp("2024-02-29")


def test_momentum_window_matches_spec_example():
    cal = _cal()
    start, end = momentum_window("2026-08-31", cal)
    assert end == pd.Timestamp("2026-07-31")
    assert start == pd.Timestamp("2025-07-31")


# --------------------------------------------------------------------------- selection
def test_zscore_winsor_and_score():
    x = pd.Series([0.0, 1.0, 2.0, 3.0, 100.0])
    z = S.zscore(x)
    assert z.max() <= 3.0 and z.min() >= -3.0
    sc = S.momentum_score(pd.Series([2.0, 0.0, -1.0, -3.0]))
    assert np.allclose(sc.values, [3.0, 1.0, 0.5, 0.25])


def test_target_count_rounding():
    assert S.target_count(280) == 56
    assert S.target_count(283) == 57            # 56.6 -> 57
    assert S.target_count(282) == 56            # 56.4 -> 56


def test_buffer_prefers_current_constituents_within_120pct():
    n = 100
    codes = [f"{i:06d}" for i in range(n)]
    ram = np.linspace(5, -5, n)                 # code 0 best ... code 99 worst
    df = pd.DataFrame({"code": codes, "RAM": ram, "eligible": True})
    # target 20: auto top 16; current constituents ranked 17..24 eligible for retention
    current = {codes[22], codes[23], codes[50]}  # 50 is outside 120% (24) -> cannot be retained
    sel = S.select_constituents(df, current)
    chosen = set(sel[sel["selected"]]["code"])
    assert len(chosen) == 20
    assert set(codes[:16]) <= chosen
    assert codes[22] in chosen and codes[23] in chosen
    assert codes[50] not in chosen
    # remaining 2 slots filled by rank (codes 16, 17)
    assert codes[16] in chosen and codes[17] in chosen and codes[18] not in chosen


def test_non_eligible_never_selected():
    df = pd.DataFrame({"code": ["a", "b", "c", "d", "e"], "RAM": [9, 8, 7, 6, 5], "eligible": [False, True, True, True, True]})
    sel = S.select_constituents(df, set())
    assert not sel.loc[sel["code"] == "a", "selected"].iloc[0]
    assert sel.loc[sel["code"] == "a", "rank"].isna().all()


# --------------------------------------------------------------------------- weighting
def test_caps_are_respected_and_sum_to_one():
    df = pd.DataFrame({"code": list("abcdef"), "fmc": [100, 50, 20, 10, 5, 1], "momentum_score": [3, 2, 1.5, 1.2, 1.1, 1.0],
                       "csi300_weight": [0.10, 0.05, 0.02, 0.01, 0.005, 0.001]})
    w = W.compute_weights(df, cap_absolute=0.09, cap_multiple=3.0)
    assert abs(w["final_weight"].sum() - 1) < 1e-9
    assert (w["final_weight"] <= w["weight_cap"] + 1e-9).all()
    assert w.attrs["cap_relaxations"] >= 1          # 9%+15%+6%+3%+1.5%+0.3% < 100% -> relaxed


def test_cap_binding_case():
    df = pd.DataFrame({"code": list("abcdefghijklmnopqrst"), "fmc": [100] + [1] * 19, "momentum_score": [1.0] * 20,
                       "csi300_weight": [0.5] + [0.5 / 19] * 19})
    w = W.compute_weights(df, cap_absolute=0.09, cap_multiple=3.0)
    assert abs(w["final_weight"].sum() - 1) < 1e-9
    assert w.loc[w["code"] == "a", "final_weight"].iloc[0] <= 0.09 + 1e-9


# --------------------------------------------------------------------------- financials
def _income(rows):
    df = pd.DataFrame(rows, columns=["REPORT_DATE", "NOTICE_DATE", "NETPROFIT", "CONTINUED_NETPROFIT"])
    df["code"] = "000001"
    return df


def test_quarterly_conversion_and_point_in_time():
    inc = _income([
        ("2023-03-31", "2023-04-25", 10, None), ("2023-06-30", "2023-08-20", 25, None),
        ("2023-09-30", "2023-10-25", 30, None), ("2023-12-31", "2024-03-28", 45, None),
        ("2024-03-31", "2024-04-26", 12, 12), ("2024-06-30", "2024-08-25", 20, 20),
    ])
    q = F.quarterly_table(inc)
    q = q.set_index("period_end")
    assert q.loc["2023-06-30", "quarter_ni"] == 15 and q.loc["2023-09-30", "quarter_ni"] == 5
    assert q.loc["2023-12-31", "quarter_ni"] == 15 and q.loc["2024-03-31", "quarter_ni"] == 12
    assert q.loc["2024-06-30", "quarter_ni"] == 8 and q.loc["2024-06-30", "field"] == "CONTINUED_NETPROFIT"
    # on 2024-08-24 the Q2-2024 report is not yet public -> latest is Q1-2024
    r = F.financial_viability(q.reset_index(), "2024-08-24")
    assert r.eligible and r.latest_period == pd.Timestamp("2024-03-31") and r.four_quarter_ni == 12 + 15 + 5 + 15
    r2 = F.financial_viability(q.reset_index(), "2024-08-31")
    assert r2.latest_period == pd.Timestamp("2024-06-30") and r2.quarter_ni == 8
    # too little history
    r3 = F.financial_viability(q.reset_index(), "2023-11-01")
    assert not r3.eligible and "consecutive" in r3.reason


def test_negative_quarter_fails():
    inc = _income([("2023-03-31", "2023-04-25", 10, None), ("2023-06-30", "2023-08-20", 25, None),
                   ("2023-09-30", "2023-10-25", 30, None), ("2023-12-31", "2024-03-28", 28, None)])
    q = F.quarterly_table(inc)
    r = F.financial_viability(q, "2024-04-30")
    assert not r.eligible and "latest quarter" in r.reason and r.four_quarter_ni == 28


def test_statutory_deadline_fallback():
    inc = _income([("2023-12-31", None, 40, None)])
    q = F.quarterly_table(inc)
    assert q["available_date"].iloc[0] == pd.Timestamp("2024-04-30")


# --------------------------------------------------------------------------- universe parsing
def test_parse_change_table_multi_index():
    html = """<table><tr><td>指数代码</td><td>指数简称</td><td>调出</td><td>调出</td><td>调入</td><td>调入</td></tr>
    <tr><td></td><td></td><td>股票代码</td><td>股票名称</td><td>股票代码</td><td>股票名称</td></tr>
    <tr><td>000300</td><td>沪深 300</td><td>601299</td><td>中国北车</td><td>300003</td><td>乐普医疗</td></tr>
    <tr><td>000903</td><td>中证 100</td><td>601299</td><td>中国北车</td><td>600886</td><td>国投电力</td></tr></table>"""
    adds, removes, src = U.parse_html_changes(html)
    assert adds == {"300003": "乐普医疗"} and removes == {"601299": "中国北车"} and src == "html:multi"


def test_parse_change_table_paired():
    html = """<p>沪深300指数样本股调整名单：</p><table><tr><td>调出样本</td><td></td><td>调入样本</td><td></td></tr>
    <tr><td>证券代码</td><td>证券简称</td><td>证券代码</td><td>证券简称</td></tr>
    <tr><td>000016</td><td>G康佳Ａ</td><td>000029</td><td>G深深房</td></tr>
    <tr><td>000429</td><td>G粤高速</td><td>000562</td><td>宏源证券</td></tr></table>"""
    adds, removes, src = U.parse_html_changes(html)
    assert set(removes) == {"000016", "000429"} and set(adds) == {"000029", "000562"}


def test_explicit_effective_dates():
    cal = _cal()
    d = U.explicit_effective_date("决定于2008年1月第一个交易日调整沪深300指数样本股", "2007-12-10", cal)
    assert d == pd.Timestamp("2008-01-01")          # business-day calendar in this test
    d = U.explicit_effective_date("本次调整将于2021年6月11日收盘后生效", "2021-05-28", cal)
    assert d == pd.Timestamp("2021-06-14")
    d = U.explicit_effective_date("决定于7月3日调整沪深300指数样本股", "2006-06-12", cal)
    assert d == pd.Timestamp("2006-07-03")


def test_reconstruction_detects_inconsistency():
    events = [
        U.AdjustmentEvent(1, "2020-06-01", "t", "regular", "2020-06-15", adds={"b": ""}, removes={"a": ""}),
    ]
    current = pd.DataFrame({"as_of": [pd.Timestamp("2021-01-01")] * 2, "code": ["b", "c"], "name": ["B", "C"]})
    with pytest.raises(U.UniverseError):
        U.reconstruct_membership(events, current, expected_size=2)     # no anchor -> error


# --------------------------------------------------------------------------- index engine
def _toy_panel():
    days = pd.bdate_range("2020-01-01", "2020-03-31")
    n = len(days)
    codes = ["a", "b", "c"]
    close = pd.DataFrame({"a": np.linspace(10, 12, n), "b": np.linspace(20, 18, n), "c": np.linspace(5, 6, n)}, index=days)
    panel = {"adj_close_pr": close, "adj_open_pr": close * 0.99, "adj_high_pr": close * 1.02, "adj_low_pr": close * 0.98,
             "adj_close_tr": close * 1.001, "traded": pd.DataFrame(1, index=days, columns=codes), "ret_pr": close.pct_change().fillna(0),
             "ret_tr": close.pct_change().fillna(0), "fmc": close * 1e6, "close": close}
    return days, panel


def test_engine_levels_and_removal():
    days, panel = _toy_panel()
    cal = TradingCalendar(days)
    membership = pd.DataFrame({"code": ["a", "b", "c"], "name": ["A", "B", "C"], "start_date": [days[0]] * 3,
                               "end_date": [pd.NaT, days[30], pd.NaT]})      # b leaves the parent on day 30
    cfg = {"costs": {"commission_schedule": [{"from": "2000-01-01", "rate": 0.0003}], "slippage_per_side": 0.001,
                     "transfer_fee_schedule": [], "stamp_duty_schedule": [{"from": "2000-01-01", "buy": 0.0, "sell": 0.001}]}}
    eng = IndexEngine(panel, cal, membership, cfg)
    rebs = [Rebalance(days[0], days[0], pd.Series({"a": 0.5, "b": 0.5})), Rebalance(days[40], days[40], pd.Series({"a": 0.5, "c": 0.5}))]
    res = eng.run(rebs, "pr", 1000.0)
    lev = res["levels"]
    assert lev["Close"].iloc[0] == 1000.0
    assert (lev["High"] >= lev["Close"] - 1e-9).all() and (lev["Low"] <= lev["Close"] + 1e-9).all()
    # day 30: b dropped; level continuous (no jump other than a's return)
    r_a = panel["adj_close_pr"]["a"].iloc[30] / panel["adj_close_pr"]["a"].iloc[29]
    assert abs(lev["Close"].iloc[30] / lev["Close"].iloc[29] - r_a) < 1e-9
    assert lev["n_holdings"].iloc[30] == 1
    assert len(res["turnover"]) == 2 and res["turnover"]["one_way_turnover"].iloc[1] == pytest.approx(0.5, abs=1e-9)
    # a non-member with positive weight must raise
    with pytest.raises(MembershipError):
        eng.run([Rebalance(days[0], days[0], pd.Series({"a": 0.5, "b": 0.5})), Rebalance(days[40], days[40], pd.Series({"b": 1.0}))], "pr")


def test_cost_rates_history():
    cfg = {"commission_schedule": [{"from": "2005-01-01", "rate": 0.0008}, {"from": "2013-01-01", "rate": 0.0003}],
           "slippage_per_side": 0.001, "transfer_fee_schedule": [],
           "stamp_duty_schedule": [{"from": "2007-05-30", "buy": 0.003, "sell": 0.003}, {"from": "2008-09-19", "buy": 0.0, "sell": 0.001}]}
    b, s = cost_rates(cfg, pd.Timestamp("2007-06-01"))
    assert b == pytest.approx(0.0008 + 0.001 + 0.003) and s == pytest.approx(0.0008 + 0.001 + 0.003)
    b, s = cost_rates(cfg, pd.Timestamp("2015-01-01"))
    assert b == pytest.approx(0.0013) and s == pytest.approx(0.0023)


# --------------------------------------------------------------------------- metrics & OHLC
def test_ohlc_resampling_monthly():
    days = pd.bdate_range("2020-01-01", "2020-02-28")
    ohlc = pd.DataFrame({"Open": np.arange(len(days)) + 100.0, "High": np.arange(len(days)) + 101.0,
                         "Low": np.arange(len(days)) + 99.0, "Close": np.arange(len(days)) + 100.5}, index=days)
    m = P.resample_ohlc(ohlc, "M")
    assert len(m) == 2
    jan = ohlc.loc["2020-01"]
    assert m["Open"].iloc[0] == jan["Open"].iloc[0] and m["Close"].iloc[0] == jan["Close"].iloc[-1]
    assert m["High"].iloc[0] == jan["High"].max() and m["Low"].iloc[0] == jan["Low"].min()
    assert m.index[0] == jan.index[-1]


def test_max_drawdown():
    idx = pd.bdate_range("2020-01-01", periods=6)
    lev = pd.Series([100, 120, 90, 96, 130, 125], index=idx)
    dd = M.max_drawdown(lev)
    assert dd["max_drawdown"] == pytest.approx(-0.25)
    assert dd["peak_date"] == idx[1] and dd["bottom_date"] == idx[2] and dd["recovery_date"] == idx[4]


# --------------------------------------------------------------------------- research report helpers
def test_hsmo_style_stats_and_docx(tmp_path):
    from src import research as R
    from src.docx_export import markdown_to_docx

    idx = pd.bdate_range("2020-01-01", periods=504)
    lv = pd.Series(100 * (1.0005 ** np.arange(504)), index=idx)
    st = R.hsmo_style_stats(lv, rf=0.0)
    assert st["Total_Return"] == pytest.approx(lv.iloc[-1] / 100 - 1)
    assert st["CAGR"] == pytest.approx((lv.iloc[-1] / 100) ** (252 / 504) - 1)
    assert st["Max_Drawdown"] == 0.0
    md = tmp_path / "r.md"
    md.write_text("# 标题\n\n段落 **加粗** `code`。\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n1. 第一\n2. 第二\n\n- 点\n", encoding="utf-8")
    out = markdown_to_docx(md, tmp_path / "r.docx")
    from docx import Document
    d = Document(out)
    assert len(d.tables) == 1 and d.tables[0].cell(1, 1).text == "2"
    assert any(p.text.startswith("1.") for p in d.paragraphs)
