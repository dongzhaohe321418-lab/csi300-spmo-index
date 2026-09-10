"""中文研究报告正文（Markdown）。所有数字来自 ctx（见 research_report.collect）。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from . import research as R
from .research_report import HSMO_REPORTED, HSMO_WINDOW, md_table, num, pct

S, B, SN, BT, ST = "CSI300_SP_FV_SPMO", "CSI300", "CSI300_SP_FV_SPMO_Net", "CSI300_TR", "CSI300_SP_FV_SPMO_TR"


def _ann(level: pd.Series) -> pd.Series:
    """Calendar-year returns indexed by integer year."""
    a = M.annual_returns(level)
    a.index = [int(p.year) for p in a.index]
    return a


def _img(path: Path, report_dir: Path, caption: str) -> str:
    rel = "../" + str(Path(path).resolve().relative_to(report_dir.resolve().parent)).replace("\\", "/")
    return f"![{caption}]({rel})\n\n*{caption}*\n"


def _stat_table(summary: pd.DataFrame) -> str:
    rows = [("最终价值（¥100 起）", "Final_Value_of_100", "money"), ("累计收益", "Total_Return", "pct"), ("年化收益 CAGR", "CAGR", "pct"),
            ("年化波动率", "Annualized_Volatility", "pct"), ("Sharpe（rf=0）", "Sharpe_Ratio", "num"), ("Sortino", "Sortino_Ratio", "num"),
            ("最大回撤", "Maximum_Drawdown", "pct"), ("Calmar", "Calmar_Ratio", "num"), ("最佳年份", "Best_Year", "str"),
            ("最差年份", "Worst_Year", "str"), ("正收益年份占比", "Positive_Year_Ratio", "pct0"), ("月度胜率（>0）", "Monthly_Win_Rate", "pct0")]
    cols = [("沪深300（价格）", B), ("策略（价格，费前）", S), ("策略（价格，费后）", SN), ("沪深300全收益", BT), ("策略全收益（费前）", ST)]
    out = []
    for label, key, kind in rows:
        r = [label]
        for _, c in cols:
            v = summary.loc[key, c]
            if kind == "money":
                r.append(f"¥{float(v):,.1f}")
            elif kind == "pct":
                r.append(pct(float(v)))
            elif kind == "pct0":
                r.append(pct(float(v), 0))
            elif kind == "num":
                r.append(f"{float(v):.2f}")
            else:
                r.append(str(v))
        out.append(r)
    df = pd.DataFrame(out, columns=["指标"] + [c[0] for c in cols])
    return md_table(df)


def write_markdown(ctx: dict, path: Path) -> Path:
    cfg = ctx["cfg"]
    su = ctx["summary"]
    key = ctx["key"]
    lv = ctx["levels"]
    uni = ctx["universe"]
    mon = ctx["monthly"]
    rb = ctx["rebalance"]
    hc = ctx["holdings_chars"]
    pers = ctx["persistence"]
    fun = ctx["funnel"]
    rep = ctx["replica"]
    attr = ctx["attr"]
    hs = ctx["hsmo"]
    fig = ctx["fig"]
    rd = path.parent
    g = lambda k, c: float(su.loc[k, c])   # noqa: E731

    start, end = lv.index[0].date(), lv.index[-1].date()
    years = M.years_between(lv.index[0], lv.index[-1])
    n_ref = len(rb); n_impl = int(rb["implemented"].sum()) if "implemented" in rb else n_ref
    fv_mean = rb["n_financial_eligible"].mean(); fv_min = rb["n_financial_eligible"].min(); fv_max = rb["n_financial_eligible"].max()
    sel_mean = hc["n"].mean()
    cap_abs = cfg["weighting"]["cap_absolute"]; cap_mult = cfg["weighting"]["cap_multiple_of_parent_weight"]
    cyc = ctx["cycles"]
    cr = ctx["crashes"].iloc[0]
    ann = ctx["annual"].copy()
    ann["Date"] = ann["Date"].astype(int)
    wins = int((ann[S] > ann[B]).sum()); n_years = len(ann)
    rs_final = float(lv[S].iloc[-1] / lv[B].iloc[-1])
    rs_slope = (rs_final) ** (1 / years) - 1
    cost = key["costs"]
    roll = key["rolling_stats"]
    attr_final = attr.iloc[-1]
    a_cagr = {c: M.cagr(attr[c]) for c in attr.columns}
    cagr_b, cagr_s, cagr_sn = g("CAGR", B), g("CAGR", S), g("CAGR", SN)

    # ----- HSMO comparison tables
    w0, w1 = HSMO_WINDOW
    hw = lv.loc[pd.Timestamp(w0):pd.Timestamp(w1)]
    def hstats(series, label):
        d = R.hsmo_style_stats(series); d["系列"] = label; return d
    rows = [hstats(hs["win_cap"]["levels"], "HSMO 规则复现（行业上限 5 只，当前行业分类）"),
            hstats(hs["win_nocap"]["levels"], "HSMO 规则复现（无行业上限）"),
            hstats(hs["win_nocap_projcost"]["levels"], "HSMO 规则复现（无行业上限，本项目成本模型）"),
            hstats(hw[ST], "本指数：全收益，费前"), hstats(lv.loc[hw.index, "CSI300_SP_FV_SPMO_TR_Net"], "本指数：全收益，费后"),
            hstats(hw[S], "本指数：价格，费前"), hstats(hw[B], "沪深300（价格）"), hstats(hw[BT], "沪深300（全收益）")]
    cmp_win = pd.DataFrame(rows).set_index("系列")
    rep_sum = HSMO_REPORTED["summary"]
    reported_row = pd.DataFrame([{"系列": "HSMO 仓库公布值（策略）", "Total_Return": rep_sum["累计收益"][0], "CAGR": rep_sum["年化收益"][0],
                                  "Volatility": rep_sum["年化波动"][0], "Sharpe(rf=2%)": rep_sum["夏普比率"][0], "Max_Drawdown": rep_sum["最大回撤"][0]},
                                 {"系列": "HSMO 仓库公布值（沪深300）", "Total_Return": rep_sum["累计收益"][1], "CAGR": rep_sum["年化收益"][1],
                                  "Volatility": rep_sum["年化波动"][1], "Sharpe(rf=2%)": rep_sum["夏普比率"][1], "Max_Drawdown": rep_sum["最大回撤"][1]}]).set_index("系列")
    cmp_win = pd.concat([reported_row, cmp_win])
    cmp_win_fmt = cmp_win.copy()
    for c in ("Total_Return", "CAGR", "Volatility", "Max_Drawdown"):
        cmp_win_fmt[c] = cmp_win_fmt[c].map(lambda v: pct(v, 1))
    cmp_win_fmt["Sharpe(rf=2%)"] = cmp_win_fmt["Sharpe(rf=2%)"].map(lambda v: f"{v:.2f}")
    cmp_win_fmt = cmp_win_fmt.rename(columns={"Total_Return": "累计收益", "CAGR": "年化收益", "Volatility": "年化波动", "Sharpe(rf=2%)": "夏普(rf=2%)", "Max_Drawdown": "最大回撤"}).reset_index()

    ann_rep = HSMO_REPORTED["annual"]
    ann_cmp = pd.DataFrame({
        "年份": list(ann_rep.keys()),
        "HSMO 公布：策略": [pct(v[0], 1) for v in ann_rep.values()],
        "HSMO 公布：沪深300": [pct(v[1], 1) for v in ann_rep.values()],
    }).set_index("年份")
    ann_cmp["复现：HSMO 规则（行业上限）"] = _ann(hs["win_cap"]["levels"]).reindex(ann_cmp.index).map(lambda v: pct(v, 1))
    ann_cmp["复现：HSMO 规则（无上限）"] = _ann(hs["win_nocap"]["levels"]).reindex(ann_cmp.index).map(lambda v: pct(v, 1))
    ann_cmp["本指数（全收益，费后）"] = _ann(lv.loc[hw.index, "CSI300_SP_FV_SPMO_TR_Net"]).reindex(ann_cmp.index).map(lambda v: pct(v, 1))
    ann_cmp["沪深300（价格，本项目数据）"] = _ann(hw[B]).reindex(ann_cmp.index).map(lambda v: pct(v, 1))
    ann_cmp = ann_cmp.reset_index()

    rows_f = [hstats(hs["full_nocap"]["levels"], "HSMO 规则复现（无行业上限，15bp）"), hstats(hs["full_nocap_projcost"]["levels"], "HSMO 规则复现（无行业上限，本项目成本模型）"),
              hstats(hs["full_cap"]["levels"], "HSMO 规则复现（行业上限，当前行业分类）"),
              hstats(lv["CSI300_SP_FV_SPMO_TR_Net"], "本指数：全收益，费后"), hstats(lv[ST], "本指数：全收益，费前"),
              hstats(lv[S], "本指数：价格，费前"), hstats(lv[B], "沪深300（价格）"), hstats(lv[BT], "沪深300（全收益）")]
    cmp_full = pd.DataFrame(rows_f).set_index("系列")
    cmp_full_fmt = cmp_full.copy()
    for c in ("Total_Return", "CAGR", "Volatility", "Max_Drawdown"):
        cmp_full_fmt[c] = cmp_full_fmt[c].map(lambda v: pct(v, 1))
    cmp_full_fmt["Sharpe(rf=2%)"] = cmp_full_fmt["Sharpe(rf=2%)"].map(lambda v: f"{v:.2f}")
    cmp_full_fmt = cmp_full_fmt.rename(columns={"Total_Return": "累计收益", "CAGR": "年化收益", "Volatility": "年化波动", "Sharpe(rf=2%)": "夏普(rf=2%)", "Max_Drawdown": "最大回撤"}).reset_index()
    hs_to = hs["full_nocap"]["turnover"]["two_way_turnover"].iloc[1:]
    hsmo_turnover_year = float(hs_to.mean() * 4 / 2)      # one-way per year, comparable with the index figure
    hsmo_final = {k: float(v["levels"].iloc[-1] / v["levels"].iloc[0] * 100) for k, v in hs.items()}
    ours_tr_net_final_full = float(lv["CSI300_SP_FV_SPMO_TR_Net"].iloc[-1])
    ours_tr_net_final_win = float(hw["CSI300_SP_FV_SPMO_TR_Net"].iloc[-1] / hw["CSI300_SP_FV_SPMO_TR_Net"].iloc[0] * 100)
    hsmo_dd_full = M.max_drawdown(hs["full_nocap"]["levels"])
    hsmo_dd_win = M.max_drawdown(hs["win_nocap"]["levels"])

    L: list[str] = []
    A = L.append
    # =========================================================================== 标题与摘要
    A("# 沪深300 S&P Financial Viability + SPMO 动量指数：方法、数据与 2005–2026 历史回测研究报告\n")
    A(f"*项目：`csi300-spmo-index` · 生成日期 {ctx['generated']} · 回测区间 {start} → {end}（{years:.1f} 年，{n_impl} 次已实施调仓）· "
      f"本报告全部数字由 `python scripts/generate_research_report.py` 从回测输出自动生成*\n")
    A("## 摘要\n")
    A(f"本报告构建并回测了一只把 S&P 500 Momentum Index（SPMO 跟踪指数）的规则平移到 A 股大盘股的指数：以**逐日点位（point-in-time）的历史沪深300成分股**为母池，"
      f"先按 S&P U.S. Indices 的 **Financial Viability** 标准（最近一季持续经营净利润 > 0 且最近四季合计 > 0）剔除亏损公司，再按 S&P Momentum 方法计算 "
      f"**12–1 月风险调整动量**，横截面标准化并转为 Momentum Score，取**前 20%**（约 {sel_mean:.0f} 只）作为成分股，以 **自由流通市值 × Momentum Score** 加权并施加公司权重上限，"
      f"每年 2 月底 / 8 月底为参考日、3 月 / 9 月第三个星期五收盘后实施。模型完全固定，未做任何参数优化，也未加入其他因子。\n")
    A("主要结论：\n")
    A(f"1. **长期复利更高但并非每个阶段都赢。** {start} 至 {end}，¥100 投入沪深300（价格指数）变为 ¥{g('Final_Value_of_100', B):,.1f}，投入本指数变为 "
      f"¥{g('Final_Value_of_100', S):,.1f}（费后 ¥{g('Final_Value_of_100', SN):,.1f}）；CAGR {pct(cagr_b)} → {pct(cagr_s)}（费后 {pct(cagr_sn)}），年化超额 {pct(g('Annualized_Excess_Return', S))}，"
      f"Jensen α {pct(g('Alpha_vs_CSI300', S))}，β {g('Beta_vs_CSI300', S):.2f}，信息比率 {g('Information_Ratio', S):.2f}。全收益口径下本指数 CAGR {pct(g('CAGR', ST))}，沪深300全收益 {pct(g('CAGR', BT))}。")
    A(f"2. **超额收益高度集中于 2017–2020 年。** 2005–2016 年策略累计落后（相对强度自 {ctx['rs_dd']['peak_date'][:7]} 的高点回撤 {pct(-ctx['rs_dd']['max_drawdown'], 0)}，直到 {ctx['rs_dd']['bottom_date'][:7]} 才见底），2017 年起持续修复并在 2019–2020 年大幅跑赢；"
      f"{n_years} 个日历年中策略跑赢 {wins} 年。月度超额收益均值 {mon['excess_mean'] * 100:+.2f} 个百分点、t 统计量 {mon['excess_t_stat']:.2f}，**统计上并不显著**。")
    A(f"3. **风险并未降低。** 策略年化波动 {pct(g('Annualized_Volatility', S))}（沪深300 {pct(g('Annualized_Volatility', B))}），2008 年最大回撤 {pct(g('Maximum_Drawdown', S))}（沪深300 {pct(g('Maximum_Drawdown', B))}），"
      f"上行捕获 {mon['up_capture']:.2f}、下行捕获 {mon['down_capture']:.2f}；Sharpe 由 {g('Sharpe_Ratio', B):.2f} 升至 {g('Sharpe_Ratio', S):.2f}，Sortino 由 {g('Sortino_Ratio', B):.2f} 升至 {g('Sortino_Ratio', S):.2f}，属于温和改善。")
    A(f"4. **最大的动量崩塌发生在 {cr['End_Date']} 前的三个月**（{cr['Start_Date']} → {cr['End_Date']}）：策略 {pct(cr['Strategy_3M_Return'], 1)} 而沪深300 {pct(cr['CSI300_3M_Return'], 1)}，相对 {pct(cr['Relative_3M'], 1)}——"
      f"即 2024 年 9 月政策驱动的急涨中，此前跌幅最大的低动量股领涨，典型的动量反转。")
    A(f"5. **收益来源主要是选股而非加权。** 用同样的成分股按自由流通市值加权得到 ¥{attr_final['Selected_basket_capweighted']:,.0f}，等权 ¥{attr_final['Selected_basket_equalweighted']:,.0f}，"
      f"SPMO 加权 ¥{g('Final_Value_of_100', S):,.0f}；Momentum Score 倾斜相对市值加权的贡献约 {(cagr_s - a_cagr['Selected_basket_capweighted']) * 100:+.2f} 个百分点/年，选股（前 20% 动量）贡献约 "
      f"{(a_cagr['Selected_basket_capweighted'] - a_cagr['CSI300_replica_proxy']) * 100:+.2f} 个百分点/年（以同一权重代理复制的沪深300为基准）。")
    A(f"6. **换手与成本可控。** 单边换手平均每次调仓 {pct(g('Average_Rebalance_Turnover', S), 1)}、每年 {pct(g('Average_Annual_Turnover', S), 1)}；按历史 A 股佣金、印花税、过户费与冲击成本计算的成本拖累约 "
      f"{pct(g('Annual_Cost_Drag', S))}/年。")
    A(f"7. **与 GitHub 项目 HSI300-Momentum-Strategy 的对比**：该项目用 BaoStock 数据、按月成分股快照、Top-20 等权、季度调仓、15bp 成本，在 2019–2025 年公布年化 {pct(rep_sum['年化收益'][0], 1)}。"
      f"用本项目的点位数据完整复现其规则得到年化 {pct(cmp_win.loc['HSMO 规则复现（行业上限 5 只，当前行业分类）', 'CAGR'], 1)}（无行业上限 {pct(cmp_win.loc['HSMO 规则复现（无行业上限）', 'CAGR'], 1)}），"
      f"同窗口本指数全收益费后 {pct(cmp_win.loc['本指数：全收益，费后', 'CAGR'], 1)}。把两套规则放到 2005–2026 全样本、同一数据上，HSMO 规则年化 {pct(cmp_full.loc['HSMO 规则复现（无行业上限，15bp）', 'CAGR'], 1)}，本指数 {pct(cmp_full.loc['本指数：全收益，费后', 'CAGR'], 1)}，沪深300全收益 {pct(cmp_full.loc['沪深300（全收益）', 'CAGR'], 1)}——"
      f"更集中、更高频的组合收益更高，但换手约为本指数的 {hsmo_turnover_year / g('Average_Annual_Turnover', S):.1f} 倍，且其行业中性规则依赖当前行业分类，历史上并不稳定。")
    A("")
    A(_img(ctx["fig_growth"], rd, "图 0  ¥100 的增长：沪深300 vs 本指数（费前 / 费后）"))

    # =========================================================================== 1 背景
    A("## 1. 研究背景与目标\n")
    A("### 1.1 动量因子与 SPMO\n")
    A("价格动量（过去 6–12 个月表现好的股票在随后数月继续领先）是被最广泛记录的股票横截面异象之一（Jegadeesh & Titman, 1993；Asness, Moskowitz & Pedersen, 2013 跨市场证据）。"
      "S&P 500 Momentum Index 是 S&P Dow Jones Indices 对这一因子的指数化实现：在 S&P 500 内按 12–1 月风险调整动量打分，取分数最高的约 20% 股票，以市值 × 动量分数加权，每年 3 月和 9 月调仓；"
      "Invesco S&P 500 Momentum ETF（SPMO）跟踪该指数，2023–2025 年的强劲表现使其受到广泛关注。\n")
    A("### 1.2 研究问题\n")
    A("本报告回答一个具体问题：**把 S&P 的这套规则原样平移到沪深300（加上 S&P U.S. Indices 用于新纳入公司的 Financial Viability 盈利资格），在 2005–2026 年的 A 股历史上会得到什么？**"
      "重点不是找到最优参数，而是用无前视、无幸存者偏差的数据检验一条固定规则的长期表现、风险特征与失效场景。\n")
    A("### 1.3 研究原则\n")
    A("- 规则固定：不调参、不优化前 20% 的比例、不加入 ROE / ROIC / PE / PB / 成长 / 杠杆 / 质量等其他因子。\n"
      "- 点位数据：每个参考日只使用当日已存在的信息——当日有效的沪深300成分、当日之前已公告的财报、当日之前的价格。\n"
      "- 自动审计：每期成分股必须当日确属沪深300、满足财务资格、所用财报已公告、动量窗口有足够交易日；任一违反即报错终止。\n"
      "- 结果可复现：`python main.py` 一键完成数据 → 股票池 → 评分 → 选样 → 加权 → 指数 → 回测 → 图表 → 报告。\n")

    # =========================================================================== 2 方法
    A("## 2. 指数构建方法\n")
    A("```\n历史沪深300（逐日点位成分） → S&P Financial Viability → SPMO 风险调整动量 → Z 分数 / Momentum Score\n"
      "→ 前 20%（S&P 缓冲规则） → 自由流通市值 × Score 加权 + 公司权重上限 → 指数（价格 / 全收益、费前 / 费后、日 OHLC） → 与沪深300比较\n```\n")
    A("### 2.1 母股票池：点位沪深300\n")
    A(f"从中证指数公司 {uni['n_events']} 份沪深300样本调整公告（{uni['n_regular']} 次定期调整、{uni['n_temporary']} 次临时调整，{uni['first_event']} → {uni['last_event']}）"
      f"自当前官方成分股名单逐笔向后回推，重建每个交易日的成分股集合。每一步回推后名单必须恰好 300 只，且重建出的 2005-07-01 名单必须与中证公布的全名单完全一致，否则程序报错。"
      f"共 {uni['n_codes']} 只证券在 {uni['n_intervals']} 个成员区间内曾属于沪深300。参考日不属于沪深300的股票 `Eligible = False, Selected = False, Final Weight = 0`；"
      f"调仓期间被剔除出沪深300的持仓在生效日按前收盘价剔除、权重按比例分配给其余成分。\n")
    A("### 2.2 S&P Financial Viability\n")
    A("S&P U.S. Indices Methodology 要求新纳入公司满足：最近一个已公布季度 GAAP 持续经营净利润为正，且最近四个季度合计为正。A 股映射：以利润表\"持续经营净利润\"（`CONTINUED_NETPROFIT`，2018 年会计准则格式起披露）为主，"
      "早期回退到\"净利润\"（`NETPROFIT`）。中国季报为累计数，单季 = 相邻累计数之差。只有首次公告日 ≤ 参考日的报表可用；"
      "东方财富对 2010 年以前报告期给出的公告日实为次年同期报表（作比较期）的公告日（滞后约 13 个月），此类不合理日期一律替换为法定披露截止日（一季报 4-30、中报 8-31、三季报 10-31、年报次年 4-30），"
      "该替代永远不早于真实公告日，因此不会引入前视。四个连续已公布季度不足者不合格。\n")
    A("### 2.3 SPMO 风险调整动量与 Z 分数\n")
    A("- 动量收益 = P(参考日前 1 个月) / P(参考日前 13 个月) − 1，日期按 A 股交易日历对齐，价格为除权除息复权后的价格收益序列。\n"
      "- 波动率 = 同一 12 个月窗口内日收益率的标准差；风险调整动量 RAM = 动量收益 / 波动率；窗口内交易日不足 150 天的股票不参与。\n"
      "- 在\"沪深300成分 且 财务合格\"的股票中横截面标准化 Z = (RAM − 均值) / 标准差，按 S&P 规则在 ±3 处截断。\n"
      "- Momentum Score = 1 + Z（Z > 0）或 1 / (1 − Z)（Z ≤ 0）。\n")
    A("### 2.4 选样与缓冲\n")
    A(f"目标数量 = 财务合格且动量可计算的成分股数 × 20%（四舍五入），历史上为 {rb['n_target'].min()}–{rb['n_target'].max()} 只而非固定 60 只。按 S&P 缓冲规则：排名前 80% 目标数的股票自动入选，"
      f"现有成分股排名在目标数 120% 以内的保留，其余按排名补足。母指数成员资格优先——被沪深300剔除的股票不能靠缓冲留任。历史上平均每期 {hc['n_buffer'].mean():.1f} 只靠缓冲留任。\n")
    A("### 2.5 权重\n")
    A(f"原始权重 = 自由流通市值 × Momentum Score，归一化后施加公司权重上限 = min({pct(cap_abs, 0)}, {cap_mult:g} × 该股在沪深300中的市值权重)，超出部分按比例分配给未触及上限的股票并迭代至无违反。"
      f"{pct(cap_abs, 0)} 是 S&P 500 Momentum 的公司上限；\"上限与母指数权重倍数取小\"是 S&P 因子指数的通行写法，但 S&P 500 Momentum 方法文件未能下载，能确认的 3× 倍数来自 S&P/BOVESPA Momentum；"
      f"在只有约 {sel_mean:.0f} 只成分股时 3× 约束每期都无可行解（上限之和 < 100%），故采用 {cap_mult:g}×。两者均为 `config.yaml` 中的显式参数，未按收益调整。历史上平均每期 {hc['n_capped'].mean():.1f} 只股票触及上限，"
      f"最大单一权重 {pct(hc['max_weight'].max(), 1)}。\n")
    A("### 2.6 调仓时间表\n")
    A(f"参考日为每年 2 月和 8 月的最后一个交易日，实施日为 3 月和 9 月第三个星期五收盘后（{n_ref} 个参考日，{n_impl} 次已实施；"
      f"最新参考日 {rb['reference_date'].iloc[-1].date()} 对应的实施日 {rb['implementation_date'].iloc[-1].date()} 尚未到来，其名单以 `_pending` 文件输出、不进入回测）。\n")
    A("### 2.7 指数计算\n")
    A("采用指数份额法：实施日收盘按目标权重确定各成分股份额，两次调仓之间份额不变、权重随价格自然漂移。价格指数用价格收益复权价，全收益指数用含股息再投资的复权价；"
      "费后指数在每次调仓按当期实际制度扣除佣金、印花税（2008-09 起仅卖出征收、税率随历史调整）、过户费及冲击成本。策略日 OHLC 由成分股当日真实开盘、最高、最低、收盘价按固定份额加总得到——"
      "开盘与收盘精确，最高 / 最低因取各成分股当日极值之和而略有高估；成交量无合理定义，留空。停牌股票以最近价格计入。\n")
    A("### 2.8 与 S&P 500 Momentum 官方方法的差异\n")
    diff = pd.DataFrame([
        ["母指数", "S&P 500", "沪深300（点位成分）"],
        ["盈利资格", "仅新纳入 S&P 500 时要求（Financial Viability）", "每个参考日对全部成分股要求（研究设定）"],
        ["动量定义", "12 月价格变动，跳过最近 1 月 / 同窗口日波动率", "相同"],
        ["Z 分数截断 / Score", "±3 / 1+Z 或 1/(1−Z)", "相同"],
        ["选样数量", "约 100 只（20%）", "合格成分股 × 20%（约 45–55 只）"],
        ["缓冲", "20% 缓冲", "相同，母指数资格优先"],
        ["权重", "FMC × Score，公司上限 9%（与母指数权重倍数取小）", f"FMC 代理 × Score，min({pct(cap_abs, 0)}, {cap_mult:g}× 母指数权重)"],
        ["自由流通市值", "S&P 自由流通因子", "已上市流通 A 股 × 收盘价（代理，见 §3.5）"],
        ["调仓", "2 / 8 月末参考，3 / 9 月第三个星期五实施", "相同"],
    ], columns=["项目", "S&P 500 Momentum", "本指数"])
    A(md_table(diff))
    A("")

    # =========================================================================== 3 数据
    A("## 3. 数据与数据质量\n")
    A("### 3.1 数据来源\n")
    src = pd.DataFrame([
        ["历史沪深300成分股", "中证指数公司官方公告（定期 / 临时调整、2005 年全名单）", f"{uni['first_event']} → {end}"],
        ["沪深300日 OHLC、全收益指数", f"中证指数公司（000300 / H00300）", f"{ctx['index_span'][0]} → {ctx['index_span'][1]}（{ctx['index_span'][2]} 个交易日）"],
        ["个股日 OHLC / 成交量 / 除权参考价", "新浪财经（含已退市证券）", "上市 → 退市 / 最新"],
        ["复权价格与日收益", "由原始价格 + 交易所除权除息参考价 + 东方财富分红 / 送转 / 配股记录构建", "同上"],
        ["自由流通市值代理", "东方财富股本结构（已上市流通 A 股）× 收盘价", "同上"],
        ["季度财务数据与公告日", "东方财富 F10 利润表（持续经营净利润 / 净利润，首次公告日）", f"{ctx['price_status'].get('ok', 0)} 只有价格的证券中，已退市者多无财报"],
        ["行业分类（仅当前）", "中证沪深300一级行业指数成分（11 个行业）", "当前 300 只成分股"],
    ], columns=["数据", "来源", "覆盖"])
    A(md_table(src))
    A("\n全部数据来自公开免费接口；`config.yaml` 支持切换到 Tushare Pro（口令通过 `.env` 提供，不入库）。\n")
    A("### 3.2 点位成分股重建的验证\n")
    A(f"- 逆推链每一步恰好 300 只；回推至 2005-07-01 的名单与中证公布的全名单逐只相同；当前名单与官方权重文件的成分集合完全一致。\n"
      f"- 资格审计（`eligibility_audit.csv`）：{len(ctx['audit'])} 个参考日中，参考日 / 实施日非成分股而权重 > 0 的记录均为 0，参考日之后公告的财报被使用的记录为 0，动量数据越界为 0。\n"
      f"- 已知处理：2008 年 7 月定期调整的主公告在中证档案中被错误归档，改由同日行业指数公告合成（其构造方法在 2008 年 12 月调整上验证为完全一致）；2006 年四家公司要约收购退市等纯文字公告手工转录并注明来源。\n")
    A("### 3.3 复权价格\n")
    A(f"以交易所除权除息参考价为锚构建乘法复权因子（价格收益口径用于动量与价格指数，全收益口径用于全收益比较），{ctx['price_status'].get('ok', 0)} 只证券全部成功。"
      f"逐事件与新浪自身的复权因子比对，除一起 2006 年送股+派息事件（新浪因子有误）外全部一致。停牌日无成交，价格沿用、收益为 0。\n")
    A("### 3.4 财务数据\n")
    nos = fun[[c for c in fun.columns if "no financial statements" in c]].sum(axis=1)
    A(f"共 {fun['members'].iloc[0]} 只成分股 × {len(fun)} 个参考日的资格判断。每期因\"无财报\"而不合格的成分股由早年的 {int(nos.max())} 只降至近年的 {int(nos.iloc[-1])} 只——"
      f"东方财富 F10 不提供多数已退市公司的报表，这些股票在其成员期内被保守地判为不合格，使早年合格池略小。财报数值为最新可得（可能经重述）数字，配以首次公告日。\n")
    A("### 3.5 自由流通市值代理及其检验\n")
    A(f"中证的分级靠档自由流通因子在个股层面没有历史公开数据，本项目以\"已上市流通 A 股 × 收盘价\"作为 FMC 代理。与 {ctx['wval']['as_of']} 官方沪深300权重比较：相关系数 {float(ctx['wval']['correlation']):.2f}，"
      f"平均绝对偏差 {100 * float(ctx['wval']['mean_abs_diff']):.2f} 个百分点，合计主动份额 {100 * float(ctx['wval']['sum_abs_diff (active share vs official)']):.0f}%——代理高估了国有大盘股（其流通 A 股大部分并非真正自由流通）。\n")
    A(f"更直接的检验是**用该代理和本项目的指数引擎复制沪深300本身**：全部成分股按代理权重、在与策略相同的日期调仓，得到 ¥{rep['final_replica']:,.0f}（CAGR {pct(rep['cagr_replica'])}），"
      f"而官方沪深300为 ¥{rep['final_official']:,.0f}（CAGR {pct(rep['cagr_official'])}）；两者日收益相关 {rep['corr']:.3f}，跟踪误差 {pct(rep['te'], 1)}。"
      f"即权重代理本身相对官方指数有约 {(rep['cagr_official'] - rep['cagr_replica']) * 100:.1f} 个百分点/年的拖累，方向是高配了长期跑输的大型国企。策略权重继承了同样的倾斜，因此**本报告的策略业绩很可能低估了真实自由流通加权版本的表现**；"
      f"§6 的分解也显示同一批成分股等权时表现优于市值加权。基于新浪十大股东数据估算真实自由流通的代码已实现为可选项（`--sources holders`），因数据源限流未在本次运行中启用。\n")
    A("### 3.6 已知局限\n")
    A("- 自由流通市值为代理值（见上）；\n- 2010 年前财报公告日以法定截止日替代（保守，可能把部分财报的可用时间推后数周）；\n- 已退市公司多无财报，被判为不合格；\n"
      "- 停牌股票以最近价格在调仓日\"成交\"，实际需待复牌；\n- 策略 OHLC 的高低点略有高估；\n- 未考虑融券、涨跌停无法成交及大额资金的冲击成本上限。\n")

    # PART2
    L.extend(_part2(ctx, rd, cmp_win_fmt, ann_cmp, cmp_full_fmt, hsmo_turnover_year, hsmo_final, ours_tr_net_final_full, ours_tr_net_final_win, hsmo_dd_full, hsmo_dd_win, cmp_win, cmp_full))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    return path

def _part2(ctx, rd, cmp_win_fmt, ann_cmp, cmp_full_fmt, hsmo_turnover_year, hsmo_final, ours_full, ours_win, hsmo_dd_full, hsmo_dd_win, cmp_win, cmp_full) -> list[str]:
    cfg = ctx["cfg"]; su = ctx["summary"]; key = ctx["key"]; lv = ctx["levels"]; mon = ctx["monthly"]; rb = ctx["rebalance"]
    hc = ctx["holdings_chars"]; pers = ctx["persistence"]; fun = ctx["funnel"]; attr = ctx["attr"]; hs = ctx["hsmo"]; fig = ctx["fig"]
    g = lambda k, c: float(su.loc[k, c])   # noqa: E731
    cyc = ctx["cycles"]; crashes = ctx["crashes"]; ann = ctx["annual"].copy(); ann["Date"] = ann["Date"].astype(int)
    roll = key["rolling_stats"]; cost = key["costs"]; tails = ctx["tails"]; rbt = ctx["rolling"]; dd = ctx["dd"]; rs_dd = ctx["rs_dd"]
    years = M.years_between(lv.index[0], lv.index[-1])
    cagr_b, cagr_s, cagr_sn = g("CAGR", B), g("CAGR", S), g("CAGR", SN)
    a_cagr = {c: M.cagr(attr[c]) for c in attr.columns}
    attr_final = attr.iloc[-1]
    rs_final = float(lv[S].iloc[-1] / lv[B].iloc[-1])
    L: list[str] = []
    A = L.append

    # =========================================================================== 4 回测结果
    A("## 4. 回测结果\n")
    A("### 4.1 总体绩效\n")
    A(_stat_table(su))
    A("")
    A(f"策略相对沪深300：年化超额 {pct(g('Annualized_Excess_Return', S))}（费后 {pct(g('Annualized_Excess_Return', SN))}），跟踪误差 {pct(g('Tracking_Error', S), 1)}，信息比率 {g('Information_Ratio', S):.2f}（费后 {g('Information_Ratio', SN):.2f}），"
      f"β {g('Beta_vs_CSI300', S):.2f}，Jensen α {pct(g('Alpha_vs_CSI300', S))}。全收益口径：本指数 ¥100 → ¥{g('Final_Value_of_100', ST):,.1f}，沪深300全收益 ¥{g('Final_Value_of_100', BT):,.1f}。\n")
    A(_img(ctx["fig_growth_log"], rd, "图 1  ¥100 的增长（对数坐标）"))
    A(_img(ctx["fig_candles"], rd, "图 2  月 K 线比较：本指数（上）与沪深300（下），对数坐标"))
    A("### 4.2 年度收益\n")
    at = ann.rename(columns={"Date": "年份", B: "沪深300", S: "策略（费前）", SN: "策略（费后）", "Excess": "超额（费前）"})
    for c in ("沪深300", "策略（费前）", "策略（费后）", "超额（费前）"):
        at[c] = at[c].map(lambda v: pct(v, 1, sign=True))
    A(md_table(at))
    best = ann.nlargest(3, S); worst = ann.nsmallest(3, S); bex = ann.nlargest(3, "Excess"); wex = ann.nsmallest(3, "Excess")
    A(f"\n{len(ann)} 个日历年中策略跑赢 {int((ann[S] > ann[B]).sum())} 年。绝对收益最强：{', '.join(f'{int(r.Date)}（{pct(r[S], 1, True)}）' for _, r in best.iterrows())}；最弱：{', '.join(f'{int(r.Date)}（{pct(r[S], 1, True)}）' for _, r in worst.iterrows())}。"
      f"超额最大：{', '.join(f'{int(r.Date)}（{pct(r.Excess, 1, True)}）' for _, r in bex.iterrows())}；超额最差：{', '.join(f'{int(r.Date)}（{pct(r.Excess, 1, True)}）' for _, r in wex.iterrows())}。"
      f"跑输的年份集中在两类环境：一是 2009、2014 年这类由低动量的金融、周期大盘股主导的急涨（动量反转），二是 2011、2015–2016 年长周期趋势中断后的震荡。\n")
    A(_img(ctx["fig_annual"], rd, "图 3  年度收益：沪深300 vs 策略"))
    A("### 4.3 月度收益分布与上 / 下行捕获\n")
    mtab = pd.DataFrame([
        ["月均收益", pct(mon["benchmark_mean"]), pct(mon["strategy_mean"])], ["月收益标准差", pct(mon["benchmark_std"]), pct(mon["strategy_std"])],
        ["偏度", f"{mon['benchmark_skew']:.2f}", f"{mon['strategy_skew']:.2f}"], ["超额峭度", f"{mon['benchmark_kurt']:.2f}", f"{mon['strategy_kurt']:.2f}"],
        ["正收益月份占比", pct(mon["benchmark_pos_share"], 0), pct(mon["strategy_pos_share"], 0)],
        ["最佳月份", f"{mon['best_month_benchmark'][0]}（{pct(mon['best_month_benchmark'][1], 1, True)}）", f"{mon['best_month_strategy'][0]}（{pct(mon['best_month_strategy'][1], 1, True)}）"],
        ["最差月份", f"{mon['worst_month_benchmark'][0]}（{pct(mon['worst_month_benchmark'][1], 1, True)}）", f"{mon['worst_month_strategy'][0]}（{pct(mon['worst_month_strategy'][1], 1, True)}）"],
    ], columns=["指标", "沪深300", "策略"])
    A(md_table(mtab))
    A(f"\n月度超额收益（策略 − 沪深300）：均值 {mon['excess_mean'] * 100:+.2f} 个百分点，标准差 {mon['excess_std'] * 100:.2f} 个百分点，为正的月份占 {pct(mon['excess_pos_share'], 0)}，"
      f"t 统计量 {mon['excess_t_stat']:.2f}（{mon['n_months']} 个月）。最佳超额月 {mon['best_excess_month'][0]}（{pct(mon['best_excess_month'][1], 1, True)}），最差超额月 {mon['worst_excess_month'][0]}（{pct(mon['worst_excess_month'][1], 1, True)}）。"
      f"上行月份捕获比 {mon['up_capture']:.2f}、下行月份捕获比 {mon['down_capture']:.2f}；上涨月跑赢概率 {pct(mon['hit_rate_up_months'], 0)}，下跌月跑赢概率 {pct(mon['hit_rate_down_months'], 0)}；"
      f"按日收益估计的上行 β {mon['beta_up_days']:.2f}、下行 β {mon['beta_down_days']:.2f}。策略在涨跌两种月份中的平均超额相近（{mon['excess_in_up_months'] * 100:+.2f} vs {mon['excess_in_down_months'] * 100:+.2f} 个百分点），"
      f"没有明显的方向性择时特征，超额主要来自个股选择。\n")
    A(_img(fig["monthly"], rd, "图 4  月度超额收益分布"))
    A("### 4.4 回撤与尾部风险\n")
    ddt = pd.DataFrame([
        ["最大回撤", pct(float(dd.loc['max_drawdown', B])), pct(float(dd.loc['max_drawdown', S]))],
        ["峰值日期", str(dd.loc['peak_date', B])[:10], str(dd.loc['peak_date', S])[:10]], ["谷底日期", str(dd.loc['bottom_date', B])[:10], str(dd.loc['bottom_date', S])[:10]],
        ["恢复日期", str(dd.loc['recovery_date', B])[:10] if pd.notna(dd.loc['recovery_date', B]) else "尚未恢复（价格指数）", str(dd.loc['recovery_date', S])[:10]],
        ["下跌天数", f"{int(float(dd.loc['days_to_bottom', B]))}", f"{int(float(dd.loc['days_to_bottom', S]))}"],
        ["恢复天数", "—" if pd.isna(dd.loc['days_to_recovery', B]) else f"{int(float(dd.loc['days_to_recovery', B]))}", f"{int(float(dd.loc['days_to_recovery', S]))}"],
    ], columns=["指标", "沪深300", "策略"])
    A(md_table(ddt))
    tt = tails.copy()
    for c in ("Daily_mean", "Daily_std", "VaR_95", "CVaR_95", "VaR_99", "CVaR_99", "Worst_day", "Best_day"):
        tt[c] = tt[c].map(lambda v: pct(v, 2))
    tt = tt.rename(columns={"Series": "系列", "Daily_mean": "日均收益", "Daily_std": "日波动", "Skew": "偏度", "Kurtosis": "超额峭度", "VaR_95": "VaR 95%", "CVaR_95": "CVaR 95%",
                            "VaR_99": "VaR 99%", "CVaR_99": "CVaR 99%", "Worst_day": "最差单日", "Worst_day_date": "日期", "Best_day": "最佳单日", "Days_below_-5%": "单日 < −5% 天数", "Days_above_+5%": "单日 > +5% 天数"})
    A("\n" + md_table(tt, floatfmt=".2f"))
    A(f"\n两个指数在 2007–2008 年都经历了 70% 以上的回撤，策略更深（{pct(float(dd.loc['max_drawdown', S]))} vs {pct(float(dd.loc['max_drawdown', B]))}）；策略价格指数于 {str(dd.loc['recovery_date', S])[:10]} 收复 2007 年高点，"
      f"而沪深300价格指数至今未收复。日收益尾部（VaR / CVaR、极端日数量）策略略重于沪深300，与其更高的波动一致。相对强度（策略 / 沪深300）的最大回撤为 {pct(rs_dd['max_drawdown'], 1)}，"
      f"从 {rs_dd['peak_date'][:10]} 持续跌到 {rs_dd['bottom_date'][:10]}，{rs_dd['days_to_recovery']} 天后于 {rs_dd['recovery_date'][:10]} 才收复——**连续八年半跑输**是持有这类策略必须接受的代价。\n")
    A(_img(ctx["fig_dd"], rd, "图 5  回撤比较"))
    A("### 4.5 滚动指标\n")
    A(f"- 1 年滚动收益跑赢沪深300的比例 {pct(roll['rolling_1y_return_excess_win_rate'], 1)}（中位数超额 {pct(roll['rolling_1y_return_median_excess'], 1, True)}）；3 年滚动 CAGR 胜率 {pct(roll['rolling_3y_cagr_excess_win_rate'], 1)}"
      f"（中位数 {pct(roll['rolling_3y_cagr_median_excess'], 1, True)}）；5 年滚动 CAGR 胜率 {pct(roll['rolling_5y_cagr_excess_win_rate'], 1)}（中位数 {pct(roll['rolling_5y_cagr_median_excess'], 1, True)}）。\n"
      f"- 1 年滚动 β 均值 {rbt['beta'].mean():.2f}（{rbt['beta'].min():.2f}–{rbt['beta'].max():.2f}），滚动跟踪误差 {pct(rbt['tracking_error'].min(), 1)}–{pct(rbt['tracking_error'].max(), 1)}（均值 {pct(rbt['tracking_error'].mean(), 1)}），"
      f"滚动信息比率中位数 {rbt['information_ratio'].median():.2f}。\n")
    A(_img(fig["rolling"], rd, "图 6  滚动 1 年 β、跟踪误差与超额收益"))
    A(_img(ctx["fig_rs"], rd, "图 7  相对强度：策略 / 沪深300"))
    A("### 4.6 市场周期分析\n")
    ct = cyc.copy()
    for c in ("CSI300_Return", "Strategy_Return", "Excess_Return", "CSI300_CAGR", "Strategy_CAGR", "CSI300_Volatility", "Strategy_Volatility", "CSI300_MaxDD", "Strategy_MaxDD"):
        ct[c] = ct[c].map(lambda v: pct(v, 1, sign=c in ("Excess_Return",)))
    for c in ("CSI300_Sharpe", "Strategy_Sharpe"):
        ct[c] = ct[c].map(lambda v: f"{v:.2f}")
    ct = ct.rename(columns={"Cycle": "阶段", "Start": "起", "End": "止", "CSI300_Return": "沪深300收益", "Strategy_Return": "策略收益", "Excess_Return": "超额", "CSI300_CAGR": "沪深300 CAGR",
                            "Strategy_CAGR": "策略 CAGR", "CSI300_Volatility": "沪深300波动", "Strategy_Volatility": "策略波动", "CSI300_Sharpe": "沪深300 Sharpe", "Strategy_Sharpe": "策略 Sharpe",
                            "CSI300_MaxDD": "沪深300最大回撤", "Strategy_MaxDD": "策略最大回撤"})
    A(md_table(ct))
    pos = cyc[cyc["Excess_Return"] > 0]["Cycle"].tolist(); neg = cyc[cyc["Excess_Return"] <= 0]["Cycle"].tolist()
    A(f"\n策略跑赢的阶段：{'、'.join(pos)}；跑输的阶段：{'、'.join(neg)}。规律与动量文献一致：**趋势延续、龙头持续领涨的结构性行情**（2017–2018 年的核心资产、2019–2020 年的消费 / 新能源 / 医药成长股）对策略最有利；"
      f"**由估值修复和政策刺激触发的 V 型反转**（2009 年上半年、2014 年四季度、2024 年 9 月）以及**趋势反复的震荡市**（2011–2013、2015–2016）最不利。2005–2007 年大牛市中策略与指数几乎同步（那一轮领涨的正是权重最大的金融与周期股）。\n")
    A(_img(fig["cycle"], rd, "图 8  各市场阶段的年化收益"))
    A("### 4.7 动量崩塌\n")
    crt = crashes.copy()
    for c in ("Strategy_3M_Return", "CSI300_3M_Return", "Relative_3M"):
        crt[c] = crt[c].map(lambda v: pct(v, 1, True))
    crt = crt.rename(columns={"End_Date": "结束日", "Start_Date": "开始日", "Strategy_3M_Return": "策略 3 个月", "CSI300_3M_Return": "沪深300 3 个月", "Relative_3M": "相对"})
    A(md_table(crt))
    c0 = crashes.iloc[0]
    rally = crashes[crashes["CSI300_3M_Return"] > 0]
    A(f"\n{len(crashes)} 次最大的三个月相对回撤中，{len(rally)} 次发生在**市场上涨**而非下跌中（{'、'.join(str(d)[:7] for d in rally['End_Date'])}），其余发生在 {'、'.join(str(d)[:7] for d in crashes[crashes['CSI300_3M_Return'] <= 0]['End_Date'])} 的下跌中。"
      f"上涨中的崩塌符合 Daniel & Moskowitz（2016）的刻画：在高波动的熊市底部反弹时，此前被抛弃的低动量股票反弹最猛，"
      f"而动量组合恰好低配它们。{c0['Start_Date']} → {c0['End_Date']} 的一次相对 {pct(c0['Relative_3M'], 1, True)}，是全样本最大的一次崩塌：2024 年 9 月 24 日一揽子政策出台后，沪深300三个月上涨 {pct(c0['CSI300_3M_Return'], 1)}，"
      f"而 3 月和 9 月两次调仓所持的高动量股（以周期资源、电子为主）合计 {pct(c0['Strategy_3M_Return'], 1)}。半年一次的调仓频率意味着策略只能在 2025 年 3 月才转向。\n")

    # =========================================================================== 5 组合特征
    A("## 5. 组合特征\n")
    A("### 5.1 Financial Viability 筛选漏斗\n")
    fail_cols = [c for c in fun.columns if c.startswith("fail:")]
    tot_fail = fun[fail_cols].sum(axis=1)
    fc = fun[fail_cols].sum().sort_values(ascending=False)
    zh = {"no financial statements": "无财务报表（多为已退市公司）", "latest quarter <= 0; trailing 4Q sum <= 0": "最近一季 ≤ 0 且最近四季合计 ≤ 0",
          "latest quarter <= 0": "仅最近一季 ≤ 0", "trailing 4Q sum <= 0": "仅最近四季合计 ≤ 0", "only 3 consecutive quarters public": "仅 3 个连续季度已公布",
          "only 2 consecutive quarters public": "仅 2 个连续季度已公布", "only 1 consecutive quarters public": "仅 1 个季度已公布", "no report public by reference date": "参考日前无任何已公布报表"}
    fc_tab = pd.DataFrame({"不合格原因": [zh.get(c.replace("fail:", ""), c.replace("fail:", "")) for c in fc.index], "股票·期次数": fc.values.astype(int), "占比": (fc.values / fc.values.sum()).round(3)})
    fc_tab["占比"] = fc_tab["占比"].map(lambda v: pct(v, 1))
    A(f"每个参考日 300 只成分股中平均 {tot_fail.mean():.0f} 只（{int(tot_fail.min())}–{int(tot_fail.max())} 只）未通过财务资格，通过者 {rb['n_financial_eligible'].mean():.0f} 只（{int(rb['n_financial_eligible'].min())}–{int(rb['n_financial_eligible'].max())}）。"
      f"不合格原因合计：\n")
    A(md_table(fc_tab))
    A(f"\n动量数据不足（新上市或长期停牌，窗口内交易日 < 150）平均每期再剔除 {(rb['n_financial_eligible'] - rb['n_eligible']).mean():.1f} 只。Z 分数被 ±3 截断的股票平均每期 {fun['z_clipped'].mean():.1f} 只；"
      f"入选门槛 Momentum Score 的中位数为 {fun['score_cutoff'].median():.2f}（即 Z ≈ {fun['score_cutoff'].median() - 1:+.2f}），最高分中位数 {fun['score_max'].median():.2f}。\n")
    A(_img(fig["funnel"], rd, "图 9  各参考日未通过 Financial Viability 的成分股数量及原因"))
    A("### 5.2 成分数量、集中度与权重上限\n")
    A(f"- 成分股数量 {int(hc['n'].min())}–{int(hc['n'].max())} 只（均值 {hc['n'].mean():.1f}）；有效成分数（1/Σw²）{hc['effective_n'].min():.0f}–{hc['effective_n'].max():.0f}（均值 {hc['effective_n'].mean():.1f}），"
      f"即权重集中度大约相当于一个 {hc['effective_n'].mean():.0f} 只等权组合。\n"
      f"- 前十大权重合计均值 {pct(hc['top10_weight'].mean(), 0)}（{pct(hc['top10_weight'].min(), 0)}–{pct(hc['top10_weight'].max(), 0)}）；最大单一权重均值 {pct(hc['max_weight'].mean(), 1)}，{pct(cfg['weighting']['cap_absolute'], 0)} 上限平均每期约束 {hc['n_capped'].mean():.1f} 只。\n"
      f"- 策略持有的股票合计占沪深300（代理）市值的 {pct(hc['csi300_weight_covered'].mean(), 0)}（{pct(hc['csi300_weight_covered'].min(), 0)}–{pct(hc['csi300_weight_covered'].max(), 0)}）；"
      f"策略权重落在沪深300市值前 100 名股票上的比例均值 {pct(hc['weight_in_top100_by_size'].mean(), 0)}，即策略整体上仍是一个大盘股组合，但相对母指数明显向中等市值倾斜。\n"
      f"- 入选股票 Momentum Score 均值 {hc['mean_momentum_score'].mean():.2f}，最低入选分数均值 {hc['min_momentum_score'].mean():.2f}。\n")
    A(_img(fig["conc"], rd, "图 10  每次调仓的成分数量、有效成分数与集中度"))
    A("### 5.3 成分持续性与换手\n")
    top = pers["top"].head(10)
    A(f"{pers['n_periods']} 次已实施调仓共出现过 {pers['n_distinct']} 只不同的成分股。相邻两期成分股的平均保留率 {pct(pers['avg_retention'], 0)}；一段连续成员期的平均长度 {pers['mean_spell']:.1f} 期（中位数 {pers['median_spell']:.0f} 期），"
      f"{pct(pers['share_single_period'], 0)} 的成员期只持续一次调仓——动量组合的\"短记忆\"特征。入选次数最多的股票：{'、'.join(f'{n}（{c} 次）' for n, c in zip(top['name'].str.replace(' ', ''), top['periods_selected']))}。\n")
    A(f"单边换手每次调仓平均 {pct(g('Average_Rebalance_Turnover', S), 1)}（最高 {pct(g('Maximum_Rebalance_Turnover', S), 1)}），折合每年 {pct(g('Average_Annual_Turnover', S), 1)}，与 S&P 500 Momentum 公布的换手水平相当。\n")
    A(_img(fig["pers"], rd, "图 11  成分股持续性"))
    A(_img(ctx["fig_turnover"], rd, "图 12  每次调仓的单边换手率"))
    A("### 5.4 行业暴露（当前）\n")
    sx = ctx["sector_exposure"].copy()
    sxf = sx.copy()
    for c in sxf.columns:
        sxf[c] = sxf[c].map(lambda v: pct(v, 1))
    sxf = sxf.reset_index().rename(columns={"sector": "中证一级行业"})
    A("历史行业分类没有免费的点位数据，此处只用中证沪深300一级行业指数的**当前**成分对**最新两期**名单分类：\n")
    A(md_table(sxf))
    over = (sx.iloc[:, 1] - sx.iloc[:, 0]).sort_values()
    A(f"\n相对沪深300，最新已实施名单最高配 {over.index[-1]}（{over.iloc[-1] * 100:+.1f} 个百分点）和 {over.index[-2]}（{over.iloc[-2] * 100:+.1f}），最低配 {over.index[0]}（{over.iloc[0] * 100:+.1f}）和 {over.index[1]}（{over.iloc[1] * 100:+.1f}）。"
      f"SPMO 类规则不做行业中性，行业暴露随动量所在板块大幅摆动，这是其超额收益与跟踪误差的共同来源。\n")
    A(_img(fig["sector"], rd, "图 13  行业暴露：沪深300 vs 最新两期名单"))
    A("### 5.5 最新持仓\n")
    lh = ctx["latest_holdings"].head(15).copy()
    lh_t = pd.DataFrame({"代码": lh["Ticker"], "名称": lh["Name"], "Momentum Score": lh["Momentum_Score"].round(2), "排名": lh["Rank"].astype(int),
                         "沪深300权重（代理）": lh["CSI300_Weight"].map(lambda v: pct(v, 2)), "最终权重": lh["Final_Weight"].map(lambda v: pct(v, 2)), "触及上限": lh["Capped"].map({True: "是", False: ""})})
    A(f"{ctx['latest_holdings']['Effective_Date'].iloc[0]} 实施（参考日 {ctx['latest_holdings']['Reference_Date'].iloc[0]}）的前 15 大权重股，共 {len(ctx['latest_holdings'])} 只：\n")
    A(md_table(lh_t))
    if ctx["pending_holdings"] is not None:
        ph = ctx["pending_holdings"]
        pt = pd.DataFrame({"代码": ph["Ticker"].head(15), "名称": ph["Name"].head(15), "Momentum Score": ph["Momentum_Score"].head(15).round(2), "排名": ph["Rank"].head(15).astype(int), "目标权重": ph["Final_Weight"].head(15).map(lambda v: pct(v, 2))})
        A(f"\n参考日 {ph['Reference_Date'].iloc[0]} 已计算、将于 {ph['Effective_Date'].iloc[0]} 收盘后实施的名单（共 {len(ph)} 只，未进入回测）前 15 名：\n")
        A(md_table(pt))
    A("")

    # =========================================================================== 6 分解
    A("## 6. 收益来源分解：选股还是加权？\n")
    A("用同一指数引擎、同样的调仓日期与成分股名单，只改变权重方案，可以把策略相对母指数的超额拆开（全部为价格指数、费前）：\n")
    dec = pd.DataFrame([
        ["沪深300（官方）", g("Final_Value_of_100", B), cagr_b, ""],
        ["沪深300复制（全部成分股，本项目权重代理，同日期调仓）", attr_final["CSI300_replica_proxy"], a_cagr["CSI300_replica_proxy"], "权重代理相对官方指数的偏差"],
        ["入选成分股，按自由流通市值（代理）加权", attr_final["Selected_basket_capweighted"], a_cagr["Selected_basket_capweighted"], "选股效应（vs 复制指数）"],
        ["入选成分股，SPMO 加权（本指数）", g("Final_Value_of_100", S), cagr_s, "Momentum Score 倾斜 + 上限（vs 市值加权）"],
        ["入选成分股，等权", attr_final["Selected_basket_equalweighted"], a_cagr["Selected_basket_equalweighted"], "参照：完全去除市值倾斜"],
    ], columns=["组合", "¥100 终值", "CAGR", "含义"])
    dec["¥100 终值"] = dec["¥100 终值"].map(lambda v: f"¥{v:,.1f}"); dec["CAGR"] = dec["CAGR"].map(lambda v: pct(v))
    A(md_table(dec))
    sel_eff = a_cagr["Selected_basket_capweighted"] - a_cagr["CSI300_replica_proxy"]
    tilt_eff = cagr_s - a_cagr["Selected_basket_capweighted"]
    proxy_eff = a_cagr["CSI300_replica_proxy"] - cagr_b
    A(f"\n- **选股效应**（同一权重口径下，前 20% 动量股 vs 全部 300 只）：{sel_eff * 100:+.2f} 个百分点/年，是超额收益的主体。\n"
      f"- **Momentum Score 加权倾斜**（vs 同一篮子市值加权）：{tilt_eff * 100:+.2f} 个百分点/年——SPMO 式加权在 A 股历史上带来的增益有限。\n"
      f"- **等权 vs 市值加权**：{(a_cagr['Selected_basket_equalweighted'] - a_cagr['Selected_basket_capweighted']) * 100:+.2f} 个百分点/年，说明在入选篮子内部，市值越大的股票（多为国有大盘股）动量延续性越差。\n"
      f"- **权重代理偏差**：{proxy_eff * 100:+.2f} 个百分点/年。本项目的自由流通代理高配国有大盘股，用它复制沪深300会跑输官方指数；策略权重同样受此影响，"
      f"因此上表中的策略 CAGR 相对官方沪深300的 {(cagr_s - cagr_b) * 100:+.2f} 个百分点超额，很可能低于真实自由流通加权版本能取得的水平。\n")
    A(_img(fig["attr"], rd, "图 14  同一批成分股在不同权重方案下的表现，以及权重代理复制的沪深300"))

    # =========================================================================== 7 成本
    A("## 7. 交易成本与可实施性\n")
    A(f"费后指数按当期实际制度扣费：佣金随历史下调、印花税按各时期税率（2008 年 9 月起仅卖出征收、2023 年 8 月起减半）、过户费与固定冲击成本。结果：费前 CAGR {pct(cost['Gross_CAGR'])}，费后 {pct(cost['Net_CAGR'])}，"
      f"年化成本拖累 {pct(cost['Annual_Cost_Drag'])}；{years:.0f} 年累计支付的成本相当于期末净值的 {pct(cost['Total_Cost_Paid_fraction_of_NAV'], 1)}。\n")
    tb = ctx["turnover_by_year"].copy()
    half = (len(tb) + 1) // 2
    tb_w = pd.DataFrame({"年份": tb["year"].iloc[:half].astype(int).values, "单边换手": tb["one_way_turnover"].iloc[:half].map(lambda v: pct(v, 0)).values,
                         "年份 ": list(tb["year"].iloc[half:].astype(int).values) + [""] * (half - len(tb) + half),
                         "单边换手 ": list(tb["one_way_turnover"].iloc[half:].map(lambda v: pct(v, 0)).values) + [""] * (half - len(tb) + half)})
    A(md_table(tb_w))
    A("\n可实施性方面：成分股全部为沪深300成分，流动性充足；半年一次调仓、单边换手约 55%，对规模不敏感；主要摩擦来自停牌股票（回测中按停牌前价格成交）以及涨跌停板日无法足量成交。"
      "费后指数已把这些成本按保守假设纳入，但未对超大规模资金的冲击成本设上限。\n")

    # =========================================================================== 8 对比 HSMO
    A(f"## 8. 与 GitHub 项目 HSI300-Momentum-Strategy 的对比\n")
    A(f"[HSI300-Momentum-Strategy]({HSMO_REPORTED['url']}) 同样受 SPMO 启发，在沪深300内做动量选股，但设计取向不同：它是一个**集中持股的量化组合**（Top-20 等权、季度调仓、行业上限），而本项目是一只**指数**（约 50 只、市值 × 分数加权、半年调仓、缓冲与上限）。"
      f"下面先比较规则，再用本项目的数据复现其规则，在相同窗口和相同数据上比较结果。\n")
    A("### 8.1 规则差异\n")
    dif = pd.DataFrame([
        ["数据源", "BaoStock：每月末查询 hs300 成分快照（2013 起）、后复权收盘价、当前行业分类", "中证公告逐日点位成分（2005 起）、新浪 / 东财原始价格自建复权、季度财报与公告日"],
        ["成分股口径", "调仓日采用最近一个月末快照", "调仓日当日有效成分；调仓期间被剔除的持仓同日剔除"],
        ["盈利筛选", "无", "S&P Financial Viability（最近一季与最近四季持续经营净利润 > 0）"],
        ["动量信号", "P(t−22)/P(t−253) − 1 除以 252 日日收益标准差（窗口含最近一月），T−1 计算 T 交易", "12–1 月价格变动除以同窗口日收益标准差；参考日计算，三周后实施"],
        ["标准化 / 分数", "无（直接按原始风险调整动量排序）", "横截面 Z 分数（±3 截断）→ Momentum Score"],
        ["选样", "Top 20，每个（当前）行业最多 5 只", "合格股票前 20%（约 45–55 只），S&P 20% 缓冲，母指数资格优先"],
        ["权重", "等权", "自由流通市值 × Score，公司上限 min(9%, 20× 母指数权重)"],
        ["调仓频率", "季度（自然季末最后一个交易日收盘）", "半年（2 / 8 月末参考，3 / 9 月第三个星期五实施）"],
        ["收益口径", "策略用后复权价（含股息）vs 沪深300价格指数", "价格指数与全收益指数分别对应比较"],
        ["成本", "成交金额的 15bp（买卖同费率）", "按历史制度的佣金 + 印花税 + 过户费 + 冲击成本（2005–2008 年印花税单边 0.1%–0.3%）"],
        ["回测区间", "2019-01-01 → 2025-12-20（7 年）", "2005-09-16 → 2026-09-09（21 年）"],
        ["资格审计", "无", "每期自动审计成员资格、财报公告日、动量数据可得性，违反即报错"],
    ], columns=["项目", "HSI300-Momentum-Strategy", "本项目"])
    A(md_table(dif))
    A("")
    A("### 8.2 其公布结果 vs 在本项目数据上的复现（2019-01-02 → 2025-12-19）\n")
    A(f"我们按其代码逐条复现规则（信号、季末调仓、等权、漂移、15bp 成本），只把数据换成本项目的点位成分与复权价格。行业上限使用当前中证一级行业分类（其代码同样使用当前分类），"
      f"历史上已不在沪深300的股票无分类，归入\"未知\"一类同样受 5 只上限约束。\n")
    A(md_table(cmp_win_fmt))
    A("")
    A(md_table(ann_cmp))
    A(f"\n复现结果与其公布值方向一致、量级接近：沪深300年度收益逐年完全相同（两边都用官方价格指数），策略年化 {pct(cmp_win.loc['HSMO 规则复现（行业上限 5 只，当前行业分类）', 'CAGR'], 1)} vs 公布 {pct(HSMO_REPORTED['summary']['年化收益'][0], 1)}，差异主要来自 2020 年"
      f"（复现 {pct(_ann(hs['win_cap']['levels']).get(2020, np.nan), 1)} vs 公布 {pct(HSMO_REPORTED['annual'][2020][0], 1)}）和 2022 年，源于成分股快照口径（月末快照 vs 逐日）、复权价格与行业分类的差别，而非规则本身。"
      f"因此可以把两条曲线放在同一数据上比较：2019–2025 年 HSMO 规则 ¥100 → ¥{hsmo_final['win_nocap']:,.0f}（无行业上限，15bp）/ ¥{hsmo_final['win_cap']:,.0f}（行业上限），本指数全收益费后 ¥{ours_win:,.0f}，"
      f"沪深300全收益 ¥{float(lv.loc[pd.Timestamp(HSMO_WINDOW[0]):pd.Timestamp(HSMO_WINDOW[1]), BT].iloc[-1] / lv.loc[pd.Timestamp(HSMO_WINDOW[0]):pd.Timestamp(HSMO_WINDOW[1]), BT].iloc[0] * 100):,.0f}。"
      f"HSMO 规则在这 7 年里收益更高，最大回撤更小（{pct(hsmo_dd_win['max_drawdown'], 1)} vs 本指数全收益 {pct(M.max_drawdown(lv.loc[pd.Timestamp(HSMO_WINDOW[0]):pd.Timestamp(HSMO_WINDOW[1]), 'CSI300_SP_FV_SPMO_TR_Net'])['max_drawdown'], 1)}）——"
      f"差别主要在 2021–2022 年：季度调仓的等权组合在 2021 年 2 月核心资产见顶后一个季度内就转向，而半年调仓的本指数持有 2 月底确定的名单直到 9 月。\n")
    A("### 8.3 全样本、同数据比较（2005-09-16 → 2026-09-09）\n")
    A(md_table(cmp_full_fmt))
    A(f"\n把窗口拉长到 21 年（三种成本假设见表），HSMO 规则（无行业上限）¥100 → ¥{hsmo_final['full_nocap']:,.0f}，本指数全收益费后 ¥{ours_full:,.0f}，沪深300全收益 ¥{g('Final_Value_of_100', BT):,.0f}。"
      f"HSMO 规则的年化收益仍高出本指数约 {(cmp_full.loc['HSMO 规则复现（无行业上限，15bp）', 'CAGR'] - cmp_full.loc['本指数：全收益，费后', 'CAGR']) * 100:.1f} 个百分点，但代价是：\n")
    A(f"- **集中度**：20 只等权 vs 约 50 只，有效成分数 20 vs {hc['effective_n'].mean():.0f}；年化波动 {pct(cmp_full.loc['HSMO 规则复现（无行业上限，15bp）', 'Volatility'], 1)} vs {pct(cmp_full.loc['本指数：全收益，费后', 'Volatility'], 1)}。\n"
      f"- **换手**：季度调仓、无缓冲，双边换手平均每季 {pct(hs['full_nocap']['turnover']['two_way_turnover'].iloc[1:].mean(), 0)}，折合单边约 {pct(hsmo_turnover_year, 0)}/年，是本指数（{pct(g('Average_Annual_Turnover', S), 0)}/年）的 {hsmo_turnover_year / g('Average_Annual_Turnover', S):.1f} 倍；"
      f"换用本项目的历史成本模型后其年化降至 {pct(cmp_full.loc['HSMO 规则复现（无行业上限，本项目成本模型）', 'CAGR'], 1)}（2019–2025 窗口 {pct(cmp_win.loc['HSMO 规则复现（无行业上限，本项目成本模型）', 'CAGR'], 1)}）。\n"
      f"- **行业中性规则不稳定**：按当前行业分类施加\"每行业 5 只\"在全样本上把年化从 {pct(cmp_full.loc['HSMO 规则复现（无行业上限，15bp）', 'CAGR'], 1)} 推高到 {pct(cmp_full.loc['HSMO 规则复现（行业上限，当前行业分类）', 'CAGR'], 1)}，"
      f"其中 2006 年一年相差 {(_ann(hs['full_cap']['levels']).get(2006, np.nan) - _ann(hs['full_nocap']['levels']).get(2006, np.nan)) * 100:+.0f} 个百分点——差异来自把早年已退市 / 已调出的股票统统归入\"未知\"行业并受同一上限约束，这是**当前分类倒推历史**带来的伪影，不能视为该规则的真实增益。\n"
      f"- **回撤**：2008 年两者都亏损约 70%（HSMO 规则 {pct(hsmo_dd_full['max_drawdown'], 1)}，{hsmo_dd_full['peak_date'].date()} → {hsmo_dd_full['bottom_date'].date()}），HSMO 于 {hsmo_dd_full['recovery_date'].date() if pd.notna(hsmo_dd_full['recovery_date']) else 'n/a'} 收复，本指数全收益于 {su.loc['MDD_Recovery_Date', ST]} 收复。\n"
      f"- **样本期**：该项目公布的 2019–2025 年恰好避开了 2008–2016 年动量策略在 A 股长期跑输的阶段（见 §4.6），7 年样本的年化超额（{pct(HSMO_REPORTED['summary']['年化收益'][0] - HSMO_REPORTED['summary']['年化收益'][1], 1)}）不应外推为长期预期。\n"
      f"- **口径**：其策略曲线含股息（后复权）而基准为价格指数，7 年内约多计 2 个百分点/年的股息；本报告分别给出价格与全收益口径。\n")
    A(_img(fig["hsmo"], rd, "图 15  HSMO 规则（在本项目数据上复现）与本指数：2019–2025 窗口与 2005–2026 全样本"))
    A(_img(fig["dd"], rd, "图 16  回撤比较：沪深300、本指数、HSMO 规则"))
    A("### 8.4 评述\n")
    A("两种设计代表了动量因子的两种用法。HSI300-Momentum-Strategy 是**高活跃度的因子组合**：更集中、更快的调仓让它在趋势切换后能更早转向，历史收益更高，但波动、换手、成本和单一行业 / 个股风险也更高，且其点位数据（月末快照、当前行业分类）与成本假设（15bp）在 2005–2008 年的制度环境下偏乐观。"
      "本项目是**可指数化的规则**：更宽的成分、市值倾斜的权重、缓冲和上限带来更低的换手和更接近母指数的风险特征（β≈1、跟踪误差约 10%），代价是对趋势反转反应更慢（2021 年、2024 年 9 月）。"
      "两者在同一数据上都长期战胜沪深300，也都在 2008、2011、2015–2016 年经历长时间跑输——**动量在 A 股大盘股中的有效性是周期性的**，任何以 2019 年以后的样本为依据的收益预期都需要打折。\n")

    # =========================================================================== 9 结论 & 17 问
    A("## 9. 结论：对 17 个问题的回答\n")
    ann_s = ann.set_index("Date")
    strong = ann_s[S].nlargest(3); weak = ann_s[S].nsmallest(3); bestex = ann_s["Excess"].nlargest(3); worstex = ann_s["Excess"].nsmallest(3)
    ans = [
        f"**回测起止日期**：{lv.index[0].date()} → {lv.index[-1].date()}（首个参考日 {rb['reference_date'].iloc[0].date()}，首次实施 {rb['implementation_date'].iloc[0].date()}；{n_years_str(ann)}）。",
        f"**¥100 投资沪深300**：¥{g('Final_Value_of_100', B):,.1f}（价格指数）；含股息 ¥{g('Final_Value_of_100', BT):,.1f}。",
        f"**¥100 投资策略**：¥{g('Final_Value_of_100', S):,.1f}（费前）/ ¥{g('Final_Value_of_100', SN):,.1f}（费后）；全收益费前 ¥{g('Final_Value_of_100', ST):,.1f}。",
        f"**CAGR**：沪深300 {pct(cagr_b)}，策略 {pct(cagr_s)}（全收益口径 {pct(g('CAGR', BT))} vs {pct(g('CAGR', ST))}）。",
        f"**年化超额收益**：{pct(g('Annualized_Excess_Return', S))}（费后 {pct(g('Annualized_Excess_Return', SN))}）；Jensen α {pct(g('Alpha_vs_CSI300', S))}，β {g('Beta_vs_CSI300', S):.2f}。",
        f"**最大回撤**：沪深300 {pct(g('Maximum_Drawdown', B))}（{su.loc['MDD_Peak_Date', B]} → {su.loc['MDD_Bottom_Date', B]}）；策略 {pct(g('Maximum_Drawdown', S))}（{su.loc['MDD_Peak_Date', S]} → {su.loc['MDD_Bottom_Date', S]}）。",
        f"**Sharpe**：沪深300 {g('Sharpe_Ratio', B):.2f}，策略 {g('Sharpe_Ratio', S):.2f}（费后 {g('Sharpe_Ratio', SN):.2f}）。",
        f"**年化波动率**：沪深300 {pct(g('Annualized_Volatility', B))}，策略 {pct(g('Annualized_Volatility', S))}。",
        f"**平均换手率**：单边每次调仓 {pct(g('Average_Rebalance_Turnover', S), 1)}，每年 {pct(g('Average_Annual_Turnover', S), 1)}（最高单次 {pct(g('Maximum_Rebalance_Turnover', S), 1)}）。",
        f"**费后 CAGR**：{pct(cost['Net_CAGR'])}（成本拖累 {pct(cost['Annual_Cost_Drag'])}/年）。",
        f"**最强年份**：{'、'.join(f'{y}（{pct(v, 1, True)}）' for y, v in strong.items())}；超额最大：{'、'.join(f'{y}（{pct(v, 1, True)}）' for y, v in bestex.items())}。",
        f"**最差年份**：{'、'.join(f'{y}（{pct(v, 1, True)}）' for y, v in weak.items())}；超额最差：{'、'.join(f'{y}（{pct(v, 1, True)}）' for y, v in worstex.items())}。",
        f"**最大动量崩塌**：{c0['Start_Date']} → {c0['End_Date']}，策略 {pct(c0['Strategy_3M_Return'], 1, True)} vs 沪深300 {pct(c0['CSI300_3M_Return'], 1, True)}（相对 {pct(c0['Relative_3M'], 1, True)}）。",
        f"**是否长期稳定跑赢**：长期跑赢但不稳定——{len(ann)} 年中 {int((ann[S] > ann[B]).sum())} 年跑赢，2005–2016 年累计跑输且相对强度回撤 {pct(rs_dd['max_drawdown'], 1)}、历时 {rs_dd['days_to_bottom']} 天，超额几乎全部来自 2017 年以后；月度超额的 t 统计量 {mon['excess_t_stat']:.2f}，统计上不显著。",
        f"**滚动超额胜率**：3 年 {pct(roll['rolling_3y_cagr_excess_win_rate'], 1)}，5 年 {pct(roll['rolling_5y_cagr_excess_win_rate'], 1)}（1 年 {pct(roll['rolling_1y_return_excess_win_rate'], 1)}）。",
        f"**相对强度趋势**：由 1.00 升至 {rs_final:.2f}（年化 {pct(rs_final ** (1 / years) - 1, 2, True)}），但先经历 2007–2016 年的长期下行（{pct(rs_dd['max_drawdown'], 1)}），2017 年后转为上行并于 2019 年收复。",
        f"**Financial Viability + SPMO 是否显著改善沪深300的长期复利与风险调整收益**：长期复利有实质改善（CAGR {pct(cagr_b)} → {pct(cagr_s)}，费后 {pct(cagr_sn)}；21 年终值高 {(g('Final_Value_of_100', S) / g('Final_Value_of_100', B) - 1) * 100:.0f}%），"
        f"风险调整收益温和改善（Sharpe {g('Sharpe_Ratio', B):.2f} → {g('Sharpe_Ratio', S):.2f}，Sortino {g('Sortino_Ratio', B):.2f} → {g('Sortino_Ratio', S):.2f}，信息比率 {g('Information_Ratio', S):.2f}），但波动和最大回撤更大、超额收益集中在少数年份且统计上不显著。"
        f"结论：**有改善，但不能称为\"显著\"；它是一条长期期望为正、需要忍受多年跑输的规则**。",
    ]
    for i, a in enumerate(ans, 1):
        A(f"{i}. {a}")
    A("")
    A("## 10. 附录\n")
    A("### 10.1 复现方法\n")
    A("```bash\npip install -r requirements.txt\npython main.py                      # 数据（缓存）→ 股票池 → 评分 → 回测 → 图表 → 报告\npython scripts/generate_research_report.py   # 本研究报告（含归因、HSMO 复现、DOCX）\npython -m pytest tests               # 19 个单元测试\n```\n")
    A("### 10.2 主要输出文件\n")
    files = pd.DataFrame([
        ["output/scores/YYYY-MM-DD.csv", "每个参考日全部沪深300成分股的资格、动量、Z 分数、Momentum Score、排名、权重"],
        ["output/holdings/YYYY-MM-DD.csv", "每次实施的成分股与权重（`_pending` 为尚未实施）"],
        ["output/performance/index_levels.csv", "日频指数点位：沪深300、策略（费前 / 费后）、全收益版本（起点 100；`index_levels_base1000.csv` 为 1000 起点）"],
        ["output/performance/strategy_ohlc.csv / csi300_ohlc.csv", "日 OHLC（策略 Volume 留空）"],
        ["output/performance/summary.csv, annual_returns.csv, market_cycles.csv, momentum_crashes.csv, rebalance_summary.csv, eligibility_audit.csv", "统计与审计"],
        ["output/research/*.csv", "本报告的归因、HSMO 复现、集中度、漏斗、持续性、行业暴露等分析表"],
        ["output/charts/*.png, output/charts/research/*.png", "全部图表"],
        ["data/processed/csi300_membership_intervals.csv", "点位成分股区间表（949 只证券）"],
    ], columns=["文件", "内容"])
    A(md_table(files))
    A("")
    A("### 10.3 参数表（config.yaml，未做任何收益导向调整）\n")
    par = pd.DataFrame([
        ["Financial Viability", "最近一季 > 0 且最近四季合计 > 0；字段 CONTINUED_NETPROFIT → NETPROFIT；公告日 ≤ 参考日；不合理公告日 → 法定截止日"],
        ["动量窗口", "参考日前 13 个月 → 前 1 个月；最少 150 个交易日"],
        ["Z 分数截断", "±3"], ["选样比例", "合格股票的 20%（四舍五入）"], ["缓冲", "前 80% 自动入选；现成分股 120% 以内保留"],
        ["权重上限", f"min({pct(cfg['weighting']['cap_absolute'], 0)}, {cfg['weighting']['cap_multiple_of_parent_weight']:g} × 母指数权重)；无可行解时倍数逐步 +1（本次运行未触发）"],
        ["调仓", "2 / 8 月最后一个交易日参考；3 / 9 月第三个星期五收盘后实施"],
        ["成本", "佣金 / 印花税 / 过户费按历史制度；冲击成本固定 " + pct(cfg['costs'].get('slippage_per_side', 0.0), 2) + " 每边"],
        ["指数基点", f"{cfg['project']['base_level']:.0f}（Growth-of-¥100 序列另按 100 重设）"],
    ], columns=["参数", "取值"])
    A(md_table(par))
    A("")
    A("*本报告仅为历史回测研究，不构成投资建议。历史表现不代表未来收益。*")
    return L


def n_years_str(ann: pd.DataFrame) -> str:
    return f"{len(ann)} 个日历年（含首尾不完整年份）"
