"""arXiv 风格 LaTeX 研究报告（中文，XeLaTeX / Tectonic 编译）。

`write_latex(ctx, path)` 把 research_report.collect() 的结果写成 output/report/research_report_zh.tex；
`compile_pdf(tex)` 用 tectonic（若安装）或 latexmk -xelatex 编译。字体全部来自 TeX Live（TeX Gyre Pagella /
Heros、Fandol 宋黑），不依赖系统字体，任何有 TeX Live 2022+ 的机器都能复现。
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from . import research as R
from .research_report import HSMO_REPORTED, HSMO_WINDOW

log = logging.getLogger(__name__)
S, B, SN, BT, ST, STN = "CSI300_SP_FV_SPMO", "CSI300", "CSI300_SP_FV_SPMO_Net", "CSI300_TR", "CSI300_SP_FV_SPMO_TR", "CSI300_SP_FV_SPMO_TR_Net"

_ESC = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\^{}"}
_SYM = {"→": r"$\rightarrow$", "≤": r"$\leq$", "≥": r"$\geq$", "≈": r"$\approx$", "×": r"$\times$", "±": r"$\pm$", "α": r"$\alpha$",
        "β": r"$\beta$", "σ": r"$\sigma$", "Σ": r"$\Sigma$", "²": r"$^2$", "−": "$-$", "—": "---", "–": "--", "≠": r"$\neq$"}


def esc(s) -> str:
    """Escape data-derived text for LaTeX (prose written in this module is already LaTeX)."""
    out = []
    for ch in str(s):
        if ch in _ESC:
            out.append(_ESC[ch])
        elif ch in _SYM:
            out.append(_SYM[ch])
        else:
            out.append(ch)
    return "".join(out)


def pct(x, d: int = 2, sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    s = f"{x * 100:+.{d}f}" if sign else f"{x * 100:.{d}f}"
    return s.replace("-", "$-$") + r"\%"


def num(x, d: int = 2) -> str:
    return f"{x:,.{d}f}".replace("-", "$-$")


def yen(x, d: int = 1) -> str:
    return r"\textyen" + f"{x:,.{d}f}"


def _ann(level: pd.Series) -> pd.Series:
    a = M.annual_returns(level)
    a.index = [int(p.year) for p in a.index]
    return a


def table(df: pd.DataFrame, caption: str, label: str, colspec: str | None = None, size: str = r"\small",
          note: str | None = None, escape: bool = True, longtable: bool = False, header_escape: bool = True, fit: bool = False) -> str:
    cols = list(df.columns)
    if colspec is None:
        colspec = "l" + "r" * (len(cols) - 1)
    hdr = " & ".join((esc(c) if header_escape else str(c)) for c in cols) + r" \\"
    rows = []
    for _, r in df.iterrows():
        rows.append(" & ".join((esc(v) if escape else str(v)) for v in r.values) + r" \\")
    body = "\n".join(rows)
    if longtable:
        return "\n".join([
            r"{" + size, r"\begin{longtable}{" + colspec + "}", r"\caption{" + caption + r"}\label{" + label + r"}\\", r"\toprule", hdr, r"\midrule", r"\endfirsthead",
            r"\toprule", hdr, r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot", r"\endlastfoot", body, r"\bottomrule", r"\end{longtable}", "}", ""])
    tab = [r"\begin{tabular}{" + colspec + "}", r"\toprule", hdr, r"\midrule", body, r"\bottomrule", r"\end{tabular}"]
    if fit:
        tab = [r"\resizebox{\linewidth}{!}{%"] + tab + ["}"]
    lines = [r"\begin{table}[!htbp]", r"\centering", size, r"\caption{" + caption + r"}\label{" + label + r"}"] + tab
    if note:
        lines += [r"\begin{minipage}{0.95\linewidth}\vspace{4pt}\footnotesize " + note + r"\end{minipage}"]
    lines += [r"\end{table}", ""]
    return "\n".join(lines)


def figure(path: str, caption: str, label: str, width: str = r"\textwidth") -> str:
    return "\n".join([r"\begin{figure}[!htbp]", r"\centering", r"\includegraphics[width=" + width + "]{" + path + "}",
                      r"\caption{" + caption + r"}\label{" + label + "}", r"\end{figure}", ""])


PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.4cm,headheight=14pt]{geometry}
\usepackage{amsmath}
\usepackage{fontspec}
\usepackage{unicode-math}
\setmainfont{texgyrepagella}[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,ItalicFont=*-italic,BoldItalicFont=*-bolditalic]
\setsansfont{texgyreheros}[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold]
\setmonofont{DejaVuSansMono.ttf}[Scale=0.82]
\setmathfont{texgyrepagella-math.otf}
\usepackage{xeCJK}
\setCJKmainfont{FandolSong-Regular.otf}[BoldFont=FandolHei-Bold.otf,ItalicFont=FandolKai-Regular.otf]
\setCJKsansfont{FandolHei-Regular.otf}[BoldFont=FandolHei-Bold.otf]
\setCJKmonofont{FandolFang-Regular.otf}
\xeCJKsetup{PunctStyle=quanjiao,CJKecglue={\hskip 0.15em plus 0.05em}}
\usepackage{booktabs,longtable,array,tabularx,multirow}
\usepackage{graphicx,float}
\usepackage[section]{placeins}
\usepackage{flafter}
\usepackage[font=small,labelfont=bf,labelsep=quad]{caption}
\usepackage{xcolor}
\usepackage{enumitem}
\setlist{nosep,leftmargin=1.6em}
\usepackage{fancyhdr}
\usepackage[round,authoryear]{natbib}
\setcitestyle{aysep={}}
\usepackage[colorlinks,linkcolor={blue!50!black},citecolor={green!40!black},urlcolor={blue!60!black}]{hyperref}
\renewcommand{\abstractname}{摘\quad 要}
\renewcommand{\figurename}{图}
\renewcommand{\tablename}{表}
\renewcommand{\refname}{参考文献}
\renewcommand{\contentsname}{目\quad 录}
\renewcommand{\appendixname}{附录}
\linespread{1.18}
\setlength{\parskip}{0.35em}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\newcommand{\RAM}{\mathrm{RAM}}
\newcommand{\FMC}{\mathrm{FMC}}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small 沪深300 S\&P Financial Viability + SPMO 动量指数}
\fancyhead[R]{\small \thepage}
\renewcommand{\headrulewidth}{0.3pt}
\begin{document}
"""


def write_latex(ctx: dict, path: Path) -> Path:  # noqa: C901 - one long, linear document
    cfg = ctx["cfg"]; su = ctx["summary"]; key = ctx["key"]; lv = ctx["levels"]; uni = ctx["universe"]; mon = ctx["monthly"]
    rb = ctx["rebalance"]; hc = ctx["holdings_chars"]; pers = ctx["persistence"]; fun = ctx["funnel"]; rep = ctx["replica"]
    attr = ctx["attr"]; hs = ctx["hsmo"]; fig = ctx["fig"]; cyc = ctx["cycles"]; crashes = ctx["crashes"]; dd = ctx["dd"]; rs_dd = ctx["rs_dd"]
    roll = key["rolling_stats"]; cost = key["costs"]; tails = ctx["tails"]; rbt = ctx["rolling"]
    g = lambda k, c: float(su.loc[k, c])  # noqa: E731
    rd = path.parent
    rel = lambda p: "../" + str(Path(p).resolve().relative_to(rd.resolve().parent)).replace("\\", "/")  # noqa: E731

    start, end = lv.index[0].date(), lv.index[-1].date()
    years = M.years_between(lv.index[0], lv.index[-1])
    n_ref = len(rb); n_impl = int(rb["implemented"].sum()) if "implemented" in rb else n_ref
    sel_mean = hc["n"].mean()
    cap_abs = cfg["weighting"]["cap_absolute"]; cap_mult = cfg["weighting"]["cap_multiple_of_parent_weight"]
    ann = ctx["annual"].copy(); ann["Date"] = ann["Date"].astype(int)
    wins = int((ann[S] > ann[B]).sum()); n_years = len(ann)
    cagr_b, cagr_s, cagr_sn = g("CAGR", B), g("CAGR", S), g("CAGR", SN)
    a_cagr = {c: M.cagr(attr[c]) for c in attr.columns}; attr_final = attr.iloc[-1]
    rs_final = float(lv[S].iloc[-1] / lv[B].iloc[-1])
    c0 = crashes.iloc[0]
    fail_cols = [c for c in fun.columns if c.startswith("fail:")]; tot_fail = fun[fail_cols].sum(axis=1)
    w0, w1 = HSMO_WINDOW; hw = lv.loc[pd.Timestamp(w0):pd.Timestamp(w1)]
    hst = lambda s: R.hsmo_style_stats(s)  # noqa: E731
    rep_sum = HSMO_REPORTED["summary"]
    win_rows = [("HSMO 仓库公布值（策略）", {"Total_Return": rep_sum["累计收益"][0], "CAGR": rep_sum["年化收益"][0], "Volatility": rep_sum["年化波动"][0], "Sharpe(rf=2%)": rep_sum["夏普比率"][0], "Max_Drawdown": rep_sum["最大回撤"][0]}),
                ("HSMO 仓库公布值（沪深300）", {"Total_Return": rep_sum["累计收益"][1], "CAGR": rep_sum["年化收益"][1], "Volatility": rep_sum["年化波动"][1], "Sharpe(rf=2%)": rep_sum["夏普比率"][1], "Max_Drawdown": rep_sum["最大回撤"][1]}),
                ("复现：HSMO 规则，行业上限（当前分类）", hst(hs["win_cap"]["levels"])), ("复现：HSMO 规则，无行业上限", hst(hs["win_nocap"]["levels"])),
                ("复现：HSMO 规则，无上限，本项目成本模型", hst(hs["win_nocap_projcost"]["levels"])),
                ("本指数：全收益，费前", hst(hw[ST])), ("本指数：全收益，费后", hst(hw[STN])), ("本指数：价格，费前", hst(hw[S])),
                ("沪深300（价格）", hst(hw[B])), ("沪深300（全收益）", hst(hw[BT]))]
    full_rows = [("复现：HSMO 规则，无行业上限，15\\,bp", hst(hs["full_nocap"]["levels"])), ("复现：HSMO 规则，无上限，本项目成本模型", hst(hs["full_nocap_projcost"]["levels"])),
                 ("复现：HSMO 规则，行业上限（当前分类）", hst(hs["full_cap"]["levels"])), ("本指数：全收益，费后", hst(lv[STN])), ("本指数：全收益，费前", hst(lv[ST])),
                 ("本指数：价格，费前", hst(lv[S])), ("沪深300（价格）", hst(lv[B])), ("沪深300（全收益）", hst(lv[BT]))]
    def stats_df(rows):
        return pd.DataFrame([{"系列": n, "累计收益": pct(d["Total_Return"], 1), "年化收益": pct(d["CAGR"], 1), "年化波动": pct(d["Volatility"], 1),
                              "夏普 (rf=2\\%)": f"{d['Sharpe(rf=2%)']:.2f}", "最大回撤": pct(d["Max_Drawdown"], 1)} for n, d in rows])
    cmp_win = {n: d for n, d in win_rows}; cmp_full = {n: d for n, d in full_rows}
    K_NC = "复现：HSMO 规则，无行业上限，15\\,bp"; K_PC = "复现：HSMO 规则，无上限，本项目成本模型"; K_CAP = "复现：HSMO 规则，行业上限（当前分类）"; K_OURS = "本指数：全收益，费后"
    hs_to = hs["full_nocap"]["turnover"]["two_way_turnover"].iloc[1:]
    hsmo_turnover_year = float(hs_to.mean() * 4 / 2)
    hsmo_final = {k: float(v["levels"].iloc[-1]) for k, v in hs.items()}
    hsmo_dd_full = M.max_drawdown(hs["full_nocap"]["levels"]); hsmo_dd_win = M.max_drawdown(hs["win_nocap"]["levels"])
    ours_win = float(hw[STN].iloc[-1] / hw[STN].iloc[0] * 100)
    bt_win = float(hw[BT].iloc[-1] / hw[BT].iloc[0] * 100)

    L: list[str] = [PREAMBLE]
    A = L.append
    # ------------------------------------------------------------------ title
    A(r"\title{\LARGE\bfseries 沪深300 S\&P Financial Viability + SPMO 动量指数：\\[4pt]\Large 方法、数据与 2005--2026 年历史回测}")
    A(r"\author{csi300-spmo-index 项目\thanks{代码、数据处理流程与全部输出见 \url{https://github.com/dongzhaohe321418-lab/csi300-spmo-index}。本文全部数字由 "
      r"\texttt{python scripts/generate\_research\_report.py --latex} 自动从回测输出生成；本文仅为历史回测研究，不构成投资建议。}}")
    _d = pd.Timestamp(ctx["generated"]); A(r"\date{" + f"{_d.year} 年 {_d.month} 月 {_d.day} 日" + "}")
    A(r"\maketitle")
    A(r"\thispagestyle{empty}")
    # ------------------------------------------------------------------ abstract
    A(r"\begin{abstract}")
    A(f"本文把 S\\&P 500 Momentum Index（SPMO）的编制规则平移到 A 股大盘股，构建并回测了\\textbf{{沪深300 S\\&P Financial Viability + SPMO 动量指数}}。"
      f"指数以逐日点位（point-in-time）的历史沪深300成分股为母池，按 S\\&P U.S. Indices 的 Financial Viability 标准（最近一季及最近四季持续经营净利润为正）剔除亏损公司，"
      f"依 S\\&P Momentum 方法计算 12--1 月风险调整动量并转换为 Momentum Score，选取前 20\\%（约 {sel_mean:.0f} 只）以自由流通市值乘以 Momentum Score 加权并施加公司权重上限，每年 3 月和 9 月调仓。"
      f"规则完全固定，不做参数优化。母池由中证指数公司 {uni['n_events']} 份调样公告逐笔回推重建，财务与价格数据均只使用参考日之前已公开的信息，并由程序逐期审计。"
      f"{start} 至 {end}，指数（价格口径）年化收益 {pct(cagr_s)}，沪深300为 {pct(cagr_b)}，年化超额 {pct(g('Annualized_Excess_Return', S))}（费后 {pct(g('Annualized_Excess_Return', SN))}），"
      f"Sharpe 由 {g('Sharpe_Ratio', B):.2f} 升至 {g('Sharpe_Ratio', S):.2f}，但年化波动（{pct(g('Annualized_Volatility', S))}）与最大回撤（{pct(g('Maximum_Drawdown', S))}）均高于母指数，"
      f"月度超额收益的 $t$ 统计量仅 {mon['excess_t_stat']:.2f}。超额收益高度集中于 2017--2020 年，2007--2016 年相对强度回撤 {pct(rs_dd['max_drawdown'], 1)}；最大的动量崩塌发生在 2024 年 9--12 月的政策驱动急涨中。"
      f"归因显示超额几乎全部来自选股（{(a_cagr['Selected_basket_capweighted'] - a_cagr['CSI300_replica_proxy']) * 100:+.1f} 个百分点/年），Momentum Score 加权倾斜贡献有限；"
      f"本文使用的自由流通市值代理相对官方指数每年拖累约 {(rep['cagr_official'] - rep['cagr_replica']) * 100:.1f} 个百分点，因此报告的策略业绩偏保守。"
      f"最后，本文在相同数据上复现了 GitHub 项目 HSI300-Momentum-Strategy 的 Top-20 等权季度动量规则：其公布的 2019--2025 年年化 {pct(rep_sum['年化收益'][0], 1)} 在本文数据上复现为 "
      f"{pct(cmp_win['复现：HSMO 规则，行业上限（当前分类）']['CAGR'], 1)}；在 2005--2026 年全样本上该规则年化 {pct(cmp_full[K_NC]['CAGR'], 1)}，"
      f"高于本指数（全收益费后 {pct(cmp_full[K_OURS]['CAGR'], 1)}），但换手约为两倍，且其基于当前行业分类的行业上限在历史上是伪影。")
    A(r"\end{abstract}")
    A(r"\begin{center}\begin{minipage}{0.86\textwidth}\small\noindent\textbf{关键词：}动量因子；风险调整动量；沪深300；S\&P 500 Momentum；Financial Viability；点位数据；指数编制；动量崩塌\end{minipage}\end{center}")
    A(r"\newpage\tableofcontents\vspace{1em}")

    # ------------------------------------------------------------------ 1 introduction
    A(r"\section{引言}")
    A(r"价格动量——过去 6 至 12 个月表现较好的股票在随后数月继续领先——是被记录最多的股票横截面异象之一 \cite{jegadeesh1993,asness2013,carhart1997}。"
      r"其代价同样有充分记录：动量组合的收益分布呈显著负偏，在高波动的熊市底部反弹中会出现\emph{动量崩塌} \cite{daniel2016,barroso2015}。"
      r"S\&P Dow Jones Indices 的 S\&P 500 Momentum Index 是这一因子的指数化实现 \cite{spdji_momentum}：在 S\&P 500 内按 12--1 月风险调整动量打分，选取分数最高的约 20\% 股票，以市值乘以动量分数加权，每年 3 月与 9 月调仓；"
      r"Invesco S\&P 500 Momentum ETF（代码 SPMO）跟踪该指数。")
    A(r"本文回答一个具体问题：\textbf{把这套规则原样平移到沪深300，并叠加 S\&P U.S. Indices 用于新纳入公司的 Financial Viability 盈利资格 \cite{spdji_us}，在 2005--2026 年的 A 股历史上会得到什么？}"
      r"研究目的不是寻找最优参数，而是在无前视、无幸存者偏差的数据上检验一条\emph{固定}规则的长期表现、风险特征、收益来源与失效场景。为此，本文遵循四条原则：")
    A(r"\begin{enumerate}" "\n"
      r"\item \textbf{规则固定。}不调参、不优化前 20\% 的比例、不加入 ROE、ROIC、PE、PB、成长、杠杆、质量等任何其他因子。" "\n"
      r"\item \textbf{点位数据。}每个参考日只使用当日已存在的信息：当日有效的沪深300成分、当日之前已公告的财务报表、当日之前的价格。" "\n"
      r"\item \textbf{自动审计。}每期成分股必须在参考日与实施日确属沪深300、满足财务资格、所用财报已公告、动量窗口有足够交易日；任一违反即报错终止。" "\n"
      r"\item \textbf{可复现。}\texttt{python main.py} 一条命令完成数据获取、股票池重建、评分、选样、加权、指数计算、回测、图表与报告。" "\n"
      r"\end{enumerate}")
    A(r"与既有的 A 股动量研究相比，本文的贡献主要在数据工程与工程化验证：(i) 用中证指数公司 2005 年以来的全部沪深300调样公告逐日重建点位成分股，并以\emph{每一步恰好 300 只}和\emph{回推名单与官方 2005 年全名单完全一致}作为硬性校验；"
      r"(ii) 用交易所除权除息参考价自建复权序列并逐事件校验；(iii) 识别并修正免费财务数据源中 2010 年以前公告日期系统性滞后的问题；(iv) 用同一指数引擎做\emph{选股 vs 加权}的归因，并用权重代理复制母指数来量化代理误差；"
      r"(v) 在完全相同的数据上复现另一公开动量策略的规则，使两种设计取向的比较不受数据差异干扰。")
    A(r"下文结构：第 \ref{sec:method} 节给出指数编制方法与同 S\&P 官方方法的差异；第 \ref{sec:data} 节说明数据来源、点位重建与数据质量检验；第 \ref{sec:results} 节报告回测结果；第 \ref{sec:portfolio} 节分析组合特征；"
      r"第 \ref{sec:attribution} 节做收益来源分解；第 \ref{sec:cost} 节讨论交易成本；第 \ref{sec:hsmo} 节与 HSI300-Momentum-Strategy 比较；第 \ref{sec:robust} 节把两者的设计差异逐维度拆开，检验是否存在更好的折中；第 \ref{sec:conclusion} 节总结并逐条回答研究任务书中的 17 个问题。")

    # ------------------------------------------------------------------ 2 methodology
    A(r"\section{指数构建方法}\label{sec:method}")
    A(r"\subsection{母股票池：点位沪深300}")
    A(f"记 $U(t)$ 为交易日 $t$ 有效的沪深300成分股集合。$U(t)$ 由中证指数公司 {uni['n_events']} 份沪深300样本调整公告（{uni['n_regular']} 次定期调整、{uni['n_temporary']} 次临时调整，"
      f"生效日 {uni['first_event']} 至 {uni['last_event']}）自当前官方名单逐笔向后回推得到：每撤销一次调整，名单必须恰好回到 300 只；回推至 2005-07-01 的名单必须与中证公布的全名单逐只一致，否则程序报错终止。"
      f"共 {uni['n_codes']} 只证券在 {uni['n_intervals']} 个成员区间内曾属于沪深300。参考日 $R$ 不属于 $U(R)$ 的股票资格、入选标志和权重一律置零；调仓期间被剔除出沪深300的持仓在生效日按前收盘价剔除，权重按比例分配给其余成分。")
    A(r"\subsection{S\&P Financial Viability}")
    A(r"S\&P U.S. Indices Methodology 要求新纳入公司满足：最近一个已公布季度的 GAAP 持续经营净利润为正，且最近四个季度合计为正 \cite{spdji_us}。本文对每个参考日的全部成分股施加该条件："
      r"\begin{equation}\mathrm{NI}^{q}_{i,\,\mathrm{latest}} > 0 \quad\text{且}\quad \sum_{k=0}^{3}\mathrm{NI}^{q}_{i,\,\mathrm{latest}-k} > 0 ,\end{equation}"
      r"其中 $\mathrm{NI}^{q}$ 为单季持续经营净利润。A 股映射：以利润表\emph{持续经营净利润}（\texttt{CONTINUED\_NETPROFIT}，2018 年准则格式起披露）为主，早期回退到\emph{净利润}（\texttt{NETPROFIT}）；"
      r"中国季报为年初累计数，单季值取相邻累计数之差（一季度即累计数）。只有首次公告日 $\leq R$ 的报表可用，四个连续已公布季度不足者不合格。公告日期的处理见第 \ref{sec:fin} 节。")
    A(r"\subsection{SPMO 风险调整动量、$Z$ 分数与 Momentum Score}")
    A(r"按 S\&P Momentum Indices Methodology \cite{spdji_momentum}，对参考日 $R$、股票 $i$："
      r"\begin{align}"
      r"\mathrm{Mom}_i &= \frac{P_i(R - 1\mathrm{M})}{P_i(R - 13\mathrm{M})} - 1, &"
      r"\sigma_i &= \operatorname{StdDev}\bigl\{r_{i,d}: d \in (R-13\mathrm{M},\, R-1\mathrm{M}]\bigr\}, \\"
      r"\RAM_i &= \mathrm{Mom}_i / \sigma_i, &"
      r"Z_i &= \operatorname{clip}\!\left(\frac{\RAM_i - \overline{\RAM}}{s_{\RAM}},\, -3,\, 3\right), \\"
      r"S_i &= \begin{cases} 1 + Z_i, & Z_i > 0 \\ 1/(1 - Z_i), & Z_i \leq 0 \end{cases}"
      r"\end{align}"
      r"其中 $P$ 为除权除息复权后的价格收益序列，日期按 A 股交易日历对齐，$r_{i,d}$ 为窗口内日收益率；窗口内交易日不足 150 天的股票不参与。均值与标准差在\emph{沪深300成分且财务合格}的股票间横截面计算。")
    A(r"\subsection{选样与缓冲}")
    A(f"目标数量 $N^\\ast = \\operatorname{{round}}(0.2\\times$ 合格且动量可计算的成分股数$)$，历史上为 {rb['n_target'].min()}--{rb['n_target'].max()} 只而非固定 60 只。按 S\\&P 缓冲规则：$S_i$ 排名前 $0.8N^\\ast$ 的股票自动入选，"
      f"现有成分股排名在 $1.2N^\\ast$ 以内的保留，其余按排名补足。母指数成员资格优先——被沪深300剔除的股票不能靠缓冲留任。历史上平均每期 {hc['n_buffer'].mean():.1f} 只靠缓冲留任。")
    A(r"\subsection{权重}")
    A(f"原始权重 $\\tilde w_i = \\FMC_i S_i / \\sum_j \\FMC_j S_j$，其中 $\\FMC$ 为自由流通市值。随后施加公司权重上限"
      f"\\begin{{equation}} c_i = \\min\\bigl({cap_abs * 100:.0f}\\%,\\; {cap_mult:g}\\, w^{{P}}_i\\bigr),\\end{{equation}}"
      f"$w^P_i$ 为该股在沪深300中的市值权重（以同一代理对全部 300 只成分计算）；超出上限的权重按比例分配给未触及上限的股票并迭代至无违反。{cap_abs * 100:.0f}\\% 是 S\\&P 500 Momentum 的公司上限；"
      f"“上限与母指数权重倍数取小”是 S\\&P 因子指数的通行写法，但 S\\&P 500 Momentum 方法文件未能取得，能确认的 3 倍倍数来自 S\\&P/BOVESPA Momentum 方法 \\cite{{spdji_bovespa}}；"
      f"在只有约 {sel_mean:.0f} 只成分股时 3 倍约束每期都无可行解（上限之和小于 100\\%），故采用 {cap_mult:g} 倍。两者均为配置文件中的显式参数，未按收益调整。历史上平均每期 {hc['n_capped'].mean():.1f} 只股票触及上限。")
    A(r"\subsection{调仓时间表与指数计算}")
    A(f"参考日为每年 2 月和 8 月的最后一个交易日，实施日为 3 月和 9 月第三个星期五收盘后，共 {n_ref} 个参考日、{n_impl} 次已实施调仓；最新参考日 {rb['reference_date'].iloc[-1].date()} 对应的实施日 {rb['implementation_date'].iloc[-1].date()} 尚未到来，其名单单独输出、不进入回测。"
      r"指数采用份额法：实施日收盘按目标权重确定各成分股份额 $n_i$，两次调仓之间份额不变，$I_t = \sum_i n_i P_{i,t}$ 经除数连接使调仓日前后连续。价格指数使用价格收益复权价，全收益指数使用含股息再投资的复权价；"
      r"费后指数在每次调仓按当期实际制度扣除佣金、印花税（2008 年 9 月起仅卖出征收，税率随历史调整）、过户费与固定冲击成本。策略的日 OHLC 由成分股当日真实开、高、低、收按固定份额加总得到——开盘与收盘精确，"
      r"最高与最低因取各成分股当日极值之和而略有高估；成交量无合理定义，留空。停牌股票以最近价格计入。两指数起点均设为 1000，绩效比较以 \textyen 100 为起点重设。")
    A(r"\subsection{与 S\&P 500 Momentum 官方方法的差异}")
    diff = pd.DataFrame([
        ["母指数", "S\\&P 500", "沪深300（点位成分）"],
        ["盈利资格", "仅新纳入 S\\&P 500 时要求", "每个参考日对全部成分股要求（研究设定）"],
        ["动量定义", "12 月价格变动，跳过最近 1 月；同窗口日波动率", "相同"],
        ["$Z$ 截断 / Score", "$\\pm 3$；$1+Z$ 或 $1/(1-Z)$", "相同"],
        ["选样数量", "约 100 只（20\\%）", "合格成分股 $\\times$ 20\\%（约 45--55 只）"],
        ["缓冲", "20\\% 缓冲", "相同，母指数资格优先"],
        ["权重", "FMC $\\times$ Score，公司上限 9\\%（与母指数权重倍数取小）", f"FMC 代理 $\\times$ Score，$\\min({cap_abs*100:.0f}\\%,\\,{cap_mult:g}\\,w^P)$"],
        ["自由流通市值", "S\\&P 自由流通因子", "已上市流通 A 股 $\\times$ 收盘价（代理，见 \\ref{sec:fmc} 节）"],
        ["调仓", "2 / 8 月末参考，3 / 9 月第三个星期五实施", "相同"],
    ], columns=["项目", "S\\&P 500 Momentum", "本指数"])
    A(table(diff, "本指数与 S\\&P 500 Momentum Index 方法的对照", "tab:diff", colspec=r"@{}p{2.3cm}p{6.3cm}p{6.3cm}@{}", escape=False, header_escape=False))

    # ------------------------------------------------------------------ 3 data
    A(r"\section{数据与数据质量}\label{sec:data}")
    A(r"\subsection{数据来源}")
    src = pd.DataFrame([
        ["历史沪深300成分股", "中证指数公司官方公告（定期 / 临时调整、2005 年全名单）", f"{uni['first_event']} → {end}"],
        ["沪深300日 OHLC、全收益指数", "中证指数公司（000300 / H00300）", f"{ctx['index_span'][0]} → {ctx['index_span'][1]}，{ctx['index_span'][2]} 个交易日"],
        ["个股日 OHLC、成交量、除权参考价", "新浪财经（含已退市证券）", "上市 → 退市 / 最新"],
        ["复权价格与日收益", "原始价格 + 交易所除权除息参考价 + 东方财富分红、送转、配股记录", "同上"],
        ["自由流通市值代理", "东方财富股本结构（已上市流通 A 股）× 收盘价", "同上"],
        ["季度财务数据与公告日", "东方财富 F10 利润表（持续经营净利润 / 净利润，首次公告日）", "已退市公司多无财报"],
        ["行业分类（仅当前）", "中证沪深300一级行业指数成分（11 个行业）", "当前 300 只成分股"],
    ], columns=["数据", "来源", "覆盖"])
    A(table(src, "数据来源。全部数据来自公开免费接口；配置文件支持切换到 Tushare Pro（口令经 \\texttt{.env} 提供，不入库）。", "tab:data", colspec=r"@{}p{3.6cm}p{7.2cm}p{4.2cm}@{}"))
    A(r"\subsection{点位成分股重建的验证}")
    A(f"逆推链每一步恰好 300 只；回推至 2005-07-01 的名单与中证公布的全名单逐只相同；当前名单与官方权重文件的成分集合完全一致。资格审计覆盖 {len(ctx['audit'])} 个参考日："
      f"参考日或实施日非成分股而权重大于零的记录为 0，参考日之后公告的财报被使用的记录为 0，动量数据越界为 0。两处需要人工介入的档案缺口均有文档记录："
      f"2008 年 7 月定期调整的主公告在中证档案中被错误归档，改由同日行业指数公告合成（构造方法在 2008 年 12 月调整上验证为完全一致）；2006 年四家公司要约收购退市等纯文字公告手工转录。")
    A(r"\subsection{复权价格}")
    A(f"以交易所除权除息参考价为锚构建乘法复权因子（价格收益口径用于动量与价格指数，全收益口径用于全收益比较），{ctx['price_status'].get('ok', 0)} 只证券全部成功。"
      f"逐事件与新浪自身的复权因子比对，除一起 2006 年送股加派息事件（新浪因子有误）外全部一致。停牌日无成交，价格沿用、收益为零。")
    A(r"\subsection{财务数据与公告日期}\label{sec:fin}")
    nos = fun[[c for c in fun.columns if "no financial statements" in c]].sum(axis=1)
    A(f"东方财富对 2010 年以前报告期给出的公告日期实为次年同期报表（作为比较期）的公告日，滞后约 13 个月（随机抽样 120 只股票的中位数滞后 2003--2009 年为 13 个月，2010 年后不足 2 个月）。"
      f"此类不合理日期（滞后超过 200 天或为负）与缺失日期一律替换为法定披露截止日（一季报 4 月 30 日、中报 8 月 31 日、三季报 10 月 31 日、年报次年 4 月 30 日）。该替代永远不早于真实公告日，因而不会引入前视，"
      f"但会把部分报表的可用时间推后数周（保守）。每期因“无财报”而不合格的成分股由早年的 {int(nos.max())} 只降至近年的 {int(nos.iloc[-1])} 只——数据源不提供多数已退市公司的报表，这些股票在其成员期内被保守地判为不合格。")
    A(r"\subsection{自由流通市值代理及其检验}\label{sec:fmc}")
    A(f"中证的分级靠档自由流通因子在个股层面没有历史公开数据，本文以“已上市流通 A 股 × 收盘价”作为 $\\FMC$ 代理。与 {ctx['wval']['as_of']} 官方沪深300权重比较：相关系数 {float(ctx['wval']['correlation']):.2f}，"
      f"平均绝对偏差 {100 * float(ctx['wval']['mean_abs_diff']):.2f} 个百分点，合计主动份额 {100 * float(ctx['wval']['sum_abs_diff (active share vs official)']):.0f}\\%——代理高估了国有大盘股（其流通 A 股大部分并非真正自由流通）。")
    A(f"更直接的检验是\\textbf{{用该代理和本文的指数引擎复制沪深300本身}}：全部成分股按代理权重、在与策略相同的日期调仓，得到 {yen(rep['final_replica'], 0)}（年化 {pct(rep['cagr_replica'])}），"
      f"而官方沪深300为 {yen(rep['final_official'], 0)}（年化 {pct(rep['cagr_official'])}）；两者日收益相关 {rep['corr']:.3f}，跟踪误差 {pct(rep['te'], 1)}。即权重代理本身相对官方指数有约 {(rep['cagr_official'] - rep['cagr_replica']) * 100:.1f} 个百分点/年的拖累，"
      f"方向是高配了长期跑输的大型国企。策略权重继承了同样的倾斜，因此本文报告的策略业绩很可能\\emph{{低估}}了真实自由流通加权版本的表现（第 \\ref{{sec:attribution}} 节的分解与此一致）。"
      f"基于新浪十大股东数据估算真实自由流通的代码已实现为可选项，因数据源限流未在本次运行中启用。")
    A(r"\subsection{已知局限}")
    A(r"\begin{itemize}\item 自由流通市值为代理值（见上）；\item 2010 年前财报公告日以法定截止日替代（保守）；\item 已退市公司多无财报，被判为不合格；"
      r"\item 停牌股票以最近价格在调仓日“成交”，实际需待复牌；\item 策略 OHLC 的高低点略有高估；\item 未考虑融券、涨跌停无法成交及超大规模资金的冲击成本。\end{itemize}")

    # ------------------------------------------------------------------ 4 results
    A(r"\section{回测结果}\label{sec:results}")
    A(r"\subsection{总体绩效}")
    rows = [("最终价值（\\textyen 100 起）", "Final_Value_of_100", "money"), ("累计收益", "Total_Return", "pct"), ("年化收益", "CAGR", "pct"), ("年化波动率", "Annualized_Volatility", "pct"),
            ("Sharpe（$r_f=0$）", "Sharpe_Ratio", "num"), ("Sortino", "Sortino_Ratio", "num"), ("最大回撤", "Maximum_Drawdown", "pct"), ("Calmar", "Calmar_Ratio", "num"),
            ("最佳年份", "Best_Year", "str"), ("最差年份", "Worst_Year", "str"), ("正收益年份占比", "Positive_Year_Ratio", "pct0"), ("月度胜率", "Monthly_Win_Rate", "pct0")]
    cols = [("沪深300", B), ("策略", S), ("策略（费后）", SN), ("沪深300 TR", BT), ("策略 TR", ST)]
    out = []
    for label, k, kind in rows:
        r = [label]
        for _, c in cols:
            v = su.loc[k, c]
            r.append(yen(float(v)) if kind == "money" else pct(float(v)) if kind == "pct" else pct(float(v), 0) if kind == "pct0" else f"{float(v):.2f}" if kind == "num" else esc(v).replace("-", "$-$"))
        out.append(r)
    A(table(pd.DataFrame(out, columns=["指标"] + [c[0] for c in cols]), f"{start} 至 {end} 的总体绩效。TR 为全收益（含股息再投资）口径；“费后”按第 \\ref{{sec:cost}} 节的历史成本制度扣费。", "tab:summary",
            colspec=r"@{}lrrrrr@{}", escape=False, header_escape=False, fit=True))
    A(f"策略相对沪深300：年化超额 {pct(g('Annualized_Excess_Return', S))}（费后 {pct(g('Annualized_Excess_Return', SN))}），跟踪误差 {pct(g('Tracking_Error', S), 1)}，信息比率 {g('Information_Ratio', S):.2f}（费后 {g('Information_Ratio', SN):.2f}），"
      f"$\\beta$ {g('Beta_vs_CSI300', S):.2f}，Jensen $\\alpha$ {pct(g('Alpha_vs_CSI300', S))}。图 \\ref{{fig:growth}} 给出 \\textyen 100 的增长路径，图 \\ref{{fig:candles}} 为两指数的月 K 线。")
    A(figure(rel(ctx["fig_growth_log"]), "\\textyen 100 的增长（对数坐标）：沪深300与本指数（费前 / 费后）", "fig:growth"))
    A(figure(rel(ctx["fig_candles"]), "月 K 线比较：本指数（上）与沪深300（下），对数坐标，时间轴对齐", "fig:candles"))
    A(r"\subsection{年度收益}")
    at = pd.DataFrame({"年份": ann["Date"], "沪深300": ann[B].map(lambda v: pct(v, 1, True)), "策略": ann[S].map(lambda v: pct(v, 1, True)), "策略（费后）": ann[SN].map(lambda v: pct(v, 1, True)), "超额": ann["Excess"].map(lambda v: pct(v, 1, True))})
    A(table(at, "日历年收益（2005 年自 9 月 16 日起，2026 年至 9 月 9 日止）", "tab:annual", colspec=r"@{}lrrrr@{}", escape=False, size=r"\footnotesize"))
    best = ann.nlargest(3, S); worst = ann.nsmallest(3, S); bex = ann.nlargest(3, "Excess"); wex = ann.nsmallest(3, "Excess")
    j = lambda df_, col: "、".join(f"{int(r.Date)} 年（{pct(r[col], 1, True)}）" for _, r in df_.iterrows())  # noqa: E731
    A(f"{n_years} 个日历年中策略跑赢 {wins} 年。绝对收益最强：{j(best, S)}；最弱：{j(worst, S)}。超额最大：{j(bex, 'Excess')}；超额最差：{j(wex, 'Excess')}。"
      f"跑输年份集中在两类环境：由低动量的金融、周期大盘股主导的急涨（2009、2014 年，即动量反转），以及长周期趋势中断后的震荡（2011、2015--2016 年）。")
    A(figure(rel(ctx["fig_annual"]), "年度收益：沪深300与策略", "fig:annual", width=r"0.9\textwidth"))
    A(r"\subsection{月度收益分布与上下行捕获}")
    mtab = pd.DataFrame([
        ["月均收益", pct(mon["benchmark_mean"]), pct(mon["strategy_mean"])], ["月收益标准差", pct(mon["benchmark_std"]), pct(mon["strategy_std"])],
        ["偏度", f"{mon['benchmark_skew']:.2f}", f"{mon['strategy_skew']:.2f}"], ["超额峭度", f"{mon['benchmark_kurt']:.2f}", f"{mon['strategy_kurt']:.2f}"],
        ["正收益月份占比", pct(mon["benchmark_pos_share"], 0), pct(mon["strategy_pos_share"], 0)],
        ["最佳月份", f"{mon['best_month_benchmark'][0]}（{pct(mon['best_month_benchmark'][1], 1, True)}）", f"{mon['best_month_strategy'][0]}（{pct(mon['best_month_strategy'][1], 1, True)}）"],
        ["最差月份", f"{mon['worst_month_benchmark'][0]}（{pct(mon['worst_month_benchmark'][1], 1, True)}）", f"{mon['worst_month_strategy'][0]}（{pct(mon['worst_month_strategy'][1], 1, True)}）"],
    ], columns=["指标", "沪深300", "策略"])
    A(table(mtab, f"月度收益分布（{mon['n_months']} 个月）", "tab:monthly", colspec=r"@{}lrr@{}", escape=False))
    A(f"月度超额收益（策略减沪深300）均值 {mon['excess_mean'] * 100:+.2f} 个百分点、标准差 {mon['excess_std'] * 100:.2f} 个百分点，为正的月份占 {pct(mon['excess_pos_share'], 0)}，$t$ 统计量 {mon['excess_t_stat']:.2f}——"
      f"在常规显著性水平下不能拒绝超额收益为零的假设。上行月份捕获比 {mon['up_capture']:.2f}、下行月份捕获比 {mon['down_capture']:.2f}；上涨月跑赢概率 {pct(mon['hit_rate_up_months'], 0)}，下跌月跑赢概率 {pct(mon['hit_rate_down_months'], 0)}；"
      f"按日收益估计的上行 $\\beta$ {mon['beta_up_days']:.2f}、下行 $\\beta$ {mon['beta_down_days']:.2f}。策略在涨跌两种月份中的平均超额相近（{mon['excess_in_up_months'] * 100:+.2f} 与 {mon['excess_in_down_months'] * 100:+.2f} 个百分点），"
      f"没有明显的方向性择时特征，超额主要来自个股选择（图 \\ref{{fig:monthly}}）。".replace("-", "$-$", 0))
    A(figure(rel(fig["monthly"]), "月度超额收益分布", "fig:monthly", width=r"0.85\textwidth"))
    A(r"\subsection{回撤与尾部风险}")
    ddt = pd.DataFrame([
        ["最大回撤", pct(float(dd.loc['max_drawdown', B])), pct(float(dd.loc['max_drawdown', S]))],
        ["峰值日期", str(dd.loc['peak_date', B])[:10], str(dd.loc['peak_date', S])[:10]], ["谷底日期", str(dd.loc['bottom_date', B])[:10], str(dd.loc['bottom_date', S])[:10]],
        ["恢复日期", "尚未恢复" if pd.isna(dd.loc['recovery_date', B]) else str(dd.loc['recovery_date', B])[:10], str(dd.loc['recovery_date', S])[:10]],
        ["下跌天数", f"{int(float(dd.loc['days_to_bottom', B]))}", f"{int(float(dd.loc['days_to_bottom', S]))}"],
        ["恢复天数", "---" if pd.isna(dd.loc['days_to_recovery', B]) else f"{int(float(dd.loc['days_to_recovery', B]))}", f"{int(float(dd.loc['days_to_recovery', S]))}"],
    ], columns=["指标", "沪深300", "策略"])
    A(table(ddt, "最大回撤（价格指数）", "tab:dd", colspec=r"@{}lrr@{}", escape=False))
    tt = pd.DataFrame({"系列": ["沪深300", "策略"], "日均收益": tails["Daily_mean"].map(lambda v: pct(v, 3)), "日波动": tails["Daily_std"].map(lambda v: pct(v)), "偏度": tails["Skew"].map(lambda v: num(v)),
                       "超额峭度": tails["Kurtosis"].map(lambda v: num(v)), "VaR 95\\%": tails["VaR_95"].map(lambda v: pct(v)), "CVaR 95\\%": tails["CVaR_95"].map(lambda v: pct(v)),
                       "VaR 99\\%": tails["VaR_99"].map(lambda v: pct(v)), "CVaR 99\\%": tails["CVaR_99"].map(lambda v: pct(v)), "最差单日": tails["Worst_day"].map(lambda v: pct(v)),
                       "$<-5\\%$ 天数": tails["Days_below_-5%"].astype(int).astype(str), "$>+5\\%$ 天数": tails["Days_above_+5%"].astype(int).astype(str)})
    A(table(tt, "日收益尾部统计（历史 VaR / CVaR）", "tab:tails", colspec=r"@{}l" + "r" * 11 + "@{}", escape=False, header_escape=False, size=r"\footnotesize", fit=True))
    A(f"两个指数在 2007--2008 年都经历了 70\\% 以上的回撤，策略更深（{pct(float(dd.loc['max_drawdown', S]))} 对 {pct(float(dd.loc['max_drawdown', B]))}）；策略价格指数于 {str(dd.loc['recovery_date', S])[:10]} 收复 2007 年高点，"
      f"而沪深300价格指数至今未收复。日收益尾部（VaR、CVaR、极端日数量）策略略重于沪深300，与其更高的波动一致。相对强度（策略 / 沪深300）的最大回撤为 {pct(rs_dd['max_drawdown'], 1)}，自 {rs_dd['peak_date'][:10]} 持续至 {rs_dd['bottom_date'][:10]}，"
      f"于 {rs_dd['recovery_date'][:10]} 才收复——\\textbf{{连续八年半跑输}}是持有这类策略必须接受的代价（图 \\ref{{fig:dd}}、图 \\ref{{fig:rs}}）。")
    A(figure(rel(ctx["fig_dd"]), "回撤比较", "fig:dd"))
    A(r"\subsection{滚动指标}")
    A(f"1 年滚动收益跑赢沪深300的比例 {pct(roll['rolling_1y_return_excess_win_rate'], 1)}（中位数超额 {pct(roll['rolling_1y_return_median_excess'], 1, True)}）；3 年滚动年化胜率 {pct(roll['rolling_3y_cagr_excess_win_rate'], 1)}"
      f"（中位数 {pct(roll['rolling_3y_cagr_median_excess'], 1, True)}）；5 年滚动年化胜率 {pct(roll['rolling_5y_cagr_excess_win_rate'], 1)}（中位数 {pct(roll['rolling_5y_cagr_median_excess'], 1, True)}）。"
      f"1 年滚动 $\\beta$ 均值 {rbt['beta'].mean():.2f}（{rbt['beta'].min():.2f}--{rbt['beta'].max():.2f}），滚动跟踪误差 {pct(rbt['tracking_error'].min(), 1)}--{pct(rbt['tracking_error'].max(), 1)}（均值 {pct(rbt['tracking_error'].mean(), 1)}），"
      f"滚动信息比率中位数 {rbt['information_ratio'].median():.2f}（图 \\ref{{fig:rolling}}）。")
    A(figure(rel(fig["rolling"]), "滚动 1 年 $\\beta$、跟踪误差与超额收益", "fig:rolling"))
    A(figure(rel(ctx["fig_rs"]), "相对强度：策略 / 沪深300", "fig:rs", width=r"0.9\textwidth"))
    A(r"\subsection{市场周期分析}")
    ct = pd.DataFrame({"阶段": cyc["Cycle"], "沪深300": cyc["CSI300_Return"].map(lambda v: pct(v, 1)), "策略": cyc["Strategy_Return"].map(lambda v: pct(v, 1)), "超额": cyc["Excess_Return"].map(lambda v: pct(v, 1, True)),
                       "沪深300年化": cyc["CSI300_CAGR"].map(lambda v: pct(v, 1)), "策略年化": cyc["Strategy_CAGR"].map(lambda v: pct(v, 1)),
                       "沪深300波动": cyc["CSI300_Volatility"].map(lambda v: pct(v, 1)), "策略波动": cyc["Strategy_Volatility"].map(lambda v: pct(v, 1)),
                       "Sharpe (300)": cyc["CSI300_Sharpe"].map(lambda v: num(v)), "Sharpe (策略)": cyc["Strategy_Sharpe"].map(lambda v: num(v)),
                       "回撤 (300)": cyc["CSI300_MaxDD"].map(lambda v: pct(v, 1)), "回撤 (策略)": cyc["Strategy_MaxDD"].map(lambda v: pct(v, 1))})
    A(table(ct, "各市场阶段的表现（价格指数、费前）", "tab:cycles", colspec=r"@{}l" + "r" * 11 + "@{}", escape=False, size=r"\footnotesize", fit=True))
    pos = cyc[cyc["Excess_Return"] > 0]["Cycle"].tolist(); neg = cyc[cyc["Excess_Return"] <= 0]["Cycle"].tolist()
    A(f"策略跑赢的阶段：{'、'.join(pos)}；跑输的阶段：{'、'.join(neg)}（表 \\ref{{tab:cycles}}、图 \\ref{{fig:cycle}}）。规律与动量文献一致：\\textbf{{趋势延续、龙头持续领涨的结构性行情}}（2017--2018 年的核心资产、2019--2020 年的消费、新能源与医药成长股）对策略最有利；"
      f"\\textbf{{由估值修复和政策刺激触发的 V 型反转}}（2009 年上半年、2014 年四季度、2024 年 9 月）以及\\textbf{{趋势反复的震荡市}}（2011--2013、2015--2016 年）最不利。2005--2007 年大牛市中策略与指数几乎同步——那一轮领涨的正是权重最大的金融与周期股。")
    A(figure(rel(fig["cycle"]), "各市场阶段的年化收益（标签为策略减沪深300，单位：年化收益率的百分点）", "fig:cycle"))
    A(r"\subsection{动量崩塌}")
    crt = pd.DataFrame({"结束日": crashes["End_Date"], "开始日": crashes["Start_Date"], "策略 3 个月": crashes["Strategy_3M_Return"].map(lambda v: pct(v, 1, True)),
                        "沪深300 3 个月": crashes["CSI300_3M_Return"].map(lambda v: pct(v, 1, True)), "相对": crashes["Relative_3M"].map(lambda v: pct(v, 1, True))})
    A(table(crt, "最大的五次三个月相对回撤", "tab:crash", colspec=r"@{}llrrr@{}", escape=False))
    rally = crashes[crashes["CSI300_3M_Return"] > 0]
    A(f"{len(crashes)} 次最大的三个月相对回撤中，{len(rally)} 次发生在\\textbf{{市场上涨}}而非下跌中（{'、'.join(str(d)[:7] for d in rally['End_Date'])}），符合 \\citet{{daniel2016}} 对动量崩塌的刻画：在高波动的熊市底部反弹时，此前被抛弃的低动量股票反弹最猛，而动量组合恰好低配它们。"
      f"{c0['Start_Date']} 至 {c0['End_Date']} 的一次相对 {pct(c0['Relative_3M'], 1, True)} 是全样本最大的一次：2024 年 9 月 24 日一揽子政策出台后沪深300三个月上涨 {pct(c0['CSI300_3M_Return'], 1)}，"
      f"而 3 月和 9 月两次调仓所持的高动量股（以周期资源、电子为主）合计 {pct(c0['Strategy_3M_Return'], 1)}。半年一次的调仓频率意味着策略直到 2025 年 3 月才转向。")

    # ------------------------------------------------------------------ 5 portfolio
    A(r"\section{组合特征}\label{sec:portfolio}")
    A(r"\subsection{Financial Viability 筛选漏斗}")
    zh = {"no financial statements": "无财务报表（多为已退市公司）", "latest quarter <= 0; trailing 4Q sum <= 0": "最近一季 $\\leq 0$ 且最近四季合计 $\\leq 0$",
          "latest quarter <= 0": "仅最近一季 $\\leq 0$", "trailing 4Q sum <= 0": "仅最近四季合计 $\\leq 0$", "only 3 consecutive quarters public": "仅 3 个连续季度已公布",
          "only 2 consecutive quarters public": "仅 2 个连续季度已公布", "only 1 consecutive quarters public": "仅 1 个季度已公布", "no report public by reference date": "参考日前无任何已公布报表"}
    fc = fun[fail_cols].sum().sort_values(ascending=False)
    fc_tab = pd.DataFrame({"不合格原因": [zh.get(c.replace("fail:", ""), c.replace("fail:", "")) for c in fc.index], "股票 × 期次": fc.values.astype(int).astype(str),
                           "占比": [pct(v, 1) for v in fc.values / fc.values.sum()]})
    A(f"每个参考日 300 只成分股中平均 {tot_fail.mean():.0f} 只（{int(tot_fail.min())}--{int(tot_fail.max())} 只）未通过财务资格，通过者 {rb['n_financial_eligible'].mean():.0f} 只（{int(rb['n_financial_eligible'].min())}--{int(rb['n_financial_eligible'].max())}）。"
      f"不合格原因见表 \\ref{{tab:funnel}} 与图 \\ref{{fig:funnel}}。动量数据不足（新上市或长期停牌，窗口内交易日少于 150）平均每期再剔除 {(rb['n_financial_eligible'] - rb['n_eligible']).mean():.1f} 只。"
      f"$Z$ 分数被 $\\pm 3$ 截断的股票平均每期 {fun['z_clipped'].mean():.1f} 只；入选门槛 Momentum Score 的中位数为 {fun['score_cutoff'].median():.2f}（即 $Z \\approx {fun['score_cutoff'].median() - 1:+.2f}$），最高分中位数 {fun['score_max'].median():.2f}。")
    A(table(fc_tab, f"Financial Viability 不合格原因合计（{len(fun)} 个参考日）", "tab:funnel", colspec=r"@{}lrr@{}", escape=False))
    A(figure(rel(fig["funnel"]), "各参考日未通过 Financial Viability 的成分股数量及原因", "fig:funnel"))
    A(r"\subsection{成分数量、集中度与权重上限}")
    A(f"成分股数量 {int(hc['n'].min())}--{int(hc['n'].max())} 只（均值 {hc['n'].mean():.1f}）；有效成分数 $1/\\sum_i w_i^2$ 为 {hc['effective_n'].min():.0f}--{hc['effective_n'].max():.0f}（均值 {hc['effective_n'].mean():.1f}），"
      f"即权重集中度大约相当于一个 {hc['effective_n'].mean():.0f} 只的等权组合。前十大权重合计均值 {pct(hc['top10_weight'].mean(), 0)}（{pct(hc['top10_weight'].min(), 0)}--{pct(hc['top10_weight'].max(), 0)}）；最大单一权重均值 {pct(hc['max_weight'].mean(), 1)}，"
      f"{pct(cap_abs, 0)} 上限平均每期约束 {hc['n_capped'].mean():.1f} 只。策略持有的股票合计占沪深300（代理）市值的 {pct(hc['csi300_weight_covered'].mean(), 0)}（{pct(hc['csi300_weight_covered'].min(), 0)}--{pct(hc['csi300_weight_covered'].max(), 0)}）；"
      f"策略权重落在沪深300市值前 100 名股票上的比例均值 {pct(hc['weight_in_top100_by_size'].mean(), 0)}，即策略整体仍是大盘股组合，但相对母指数明显向中等市值倾斜（图 \\ref{{fig:conc}}）。"
      f"入选股票 Momentum Score 均值 {hc['mean_momentum_score'].mean():.2f}，最低入选分数均值 {hc['min_momentum_score'].mean():.2f}。")
    A(figure(rel(fig["conc"]), "每次调仓的成分数量、有效成分数与集中度", "fig:conc"))
    A(r"\subsection{成分持续性与换手}")
    top = pers["top"].head(10)
    A(f"{pers['n_periods']} 次已实施调仓共出现过 {pers['n_distinct']} 只不同的成分股。相邻两期成分股的平均保留率 {pct(pers['avg_retention'], 0)}；一段连续成员期的平均长度 {pers['mean_spell']:.1f} 期（中位数 {pers['median_spell']:.0f} 期），"
      f"{pct(pers['share_single_period'], 0)} 的成员期只持续一次调仓——动量组合的“短记忆”特征（图 \\ref{{fig:pers}}）。入选次数最多的股票：{'、'.join(f'{esc(n)}（{c} 次）' for n, c in zip(top['name'].str.replace(' ', ''), top['periods_selected']))}。"
      f"单边换手每次调仓平均 {pct(g('Average_Rebalance_Turnover', S), 1)}（最高 {pct(g('Maximum_Rebalance_Turnover', S), 1)}），折合每年 {pct(g('Average_Annual_Turnover', S), 1)}，与 S\\&P 500 Momentum 公布的换手水平相当。")
    A(figure(rel(fig["pers"]), "成分股持续性：入选次数最多的股票（左）与连续成员期长度分布（右）", "fig:pers"))
    A(r"\subsection{行业暴露（当前分类）}")
    sx = ctx["sector_exposure"].copy()
    sxf = pd.DataFrame({"中证一级行业": sx.index})
    for c in sx.columns:
        sxf[c.replace("CSI 300", "沪深300").replace("Strategy", "策略").replace("(pending)", "（待实施）")] = sx[c].map(lambda v: pct(v, 1)).values
    A("历史行业分类没有免费的点位数据，此处只用中证沪深300一级行业指数的\\emph{当前}成分对\\emph{最新两期}名单分类（表 \\ref{tab:sector}、图 \\ref{fig:sector}）。")
    A(table(sxf, "行业暴露：沪深300官方权重与最新两期策略名单", "tab:sector", colspec=r"@{}lrrr@{}", escape=False, header_escape=True))
    over = (sx.iloc[:, 1] - sx.iloc[:, 0]).sort_values()
    A(f"相对沪深300，最新已实施名单最高配{esc(over.index[-1])}（{over.iloc[-1] * 100:+.1f} 个百分点）和{esc(over.index[-2])}（{over.iloc[-2] * 100:+.1f}），最低配{esc(over.index[0])}（{over.iloc[0] * 100:+.1f}）和{esc(over.index[1])}（{over.iloc[1] * 100:+.1f}）。"
      f"SPMO 类规则不做行业中性，行业暴露随动量所在板块大幅摆动，这是其超额收益与跟踪误差的共同来源。".replace("+", "$+$").replace("-", "$-$"))
    A(figure(rel(fig["sector"]), "行业暴露：沪深300与最新两期名单（当前中证一级行业分类）", "fig:sector", width=r"0.9\textwidth"))
    A(r"最新持仓与待实施名单见附录 \ref{app:holdings}。")

    # ------------------------------------------------------------------ 6 attribution
    A(r"\section{收益来源分解：选股还是加权？}\label{sec:attribution}")
    A(r"用同一指数引擎、同样的调仓日期与成分股名单，只改变权重方案，可以把策略相对母指数的超额拆开（价格指数、费前；表 \ref{tab:attr}、图 \ref{fig:attr}）。")
    dec = pd.DataFrame([
        ["沪深300（官方）", yen(g("Final_Value_of_100", B)), pct(cagr_b), "---"],
        ["沪深300复制（全部成分股，本文权重代理，同日期调仓）", yen(attr_final["CSI300_replica_proxy"]), pct(a_cagr["CSI300_replica_proxy"]), "权重代理相对官方指数的偏差"],
        ["入选成分股，按自由流通市值（代理）加权", yen(attr_final["Selected_basket_capweighted"]), pct(a_cagr["Selected_basket_capweighted"]), "选股效应（相对复制指数）"],
        ["入选成分股，SPMO 加权（本指数）", yen(g("Final_Value_of_100", S)), pct(cagr_s), "Momentum Score 倾斜与上限（相对市值加权）"],
        ["入选成分股，等权", yen(attr_final["Selected_basket_equalweighted"]), pct(a_cagr["Selected_basket_equalweighted"]), "参照：完全去除市值倾斜"],
    ], columns=["组合", "\\textyen 100 终值", "年化", "含义"])
    A(table(dec, "同一批成分股在不同权重方案下的表现", "tab:attr", colspec=r"@{}p{6.4cm}rrp{4.6cm}@{}", escape=False, header_escape=False))
    sel_eff = a_cagr["Selected_basket_capweighted"] - a_cagr["CSI300_replica_proxy"]; tilt_eff = cagr_s - a_cagr["Selected_basket_capweighted"]; proxy_eff = a_cagr["CSI300_replica_proxy"] - cagr_b
    A(r"\begin{itemize}")
    A(f"\\item \\textbf{{选股效应}}（同一权重口径下，前 20\\% 动量股对全部 300 只）：{sel_eff * 100:+.2f} 个百分点/年，是超额收益的主体。".replace("+", "$+$"))
    A(f"\\item \\textbf{{Momentum Score 加权倾斜}}（相对同一篮子的市值加权）：{tilt_eff * 100:+.2f} 个百分点/年——SPMO 式加权在 A 股历史上带来的增益有限。".replace("+", "$+$"))
    A(f"\\item \\textbf{{等权对市值加权}}：{(a_cagr['Selected_basket_equalweighted'] - a_cagr['Selected_basket_capweighted']) * 100:+.2f} 个百分点/年，说明在入选篮子内部，市值越大的股票（多为国有大盘股）动量延续性越差。".replace("+", "$+$"))
    A(f"\\item \\textbf{{权重代理偏差}}：{proxy_eff * 100:+.2f} 个百分点/年。代理高配国有大盘股，用它复制沪深300会跑输官方指数；策略权重同样受此影响，因此策略相对官方沪深300 {(cagr_s - cagr_b) * 100:+.2f} 个百分点的年化超额很可能低于真实自由流通加权版本能取得的水平。".replace("+", "$+$").replace("$-$", "$-$"))
    A(r"\end{itemize}")
    A(figure(rel(fig["attr"]), "同一批成分股在三种权重方案下的表现，以及用权重代理复制的沪深300（\\textyen 100，对数坐标）", "fig:attr"))

    # ------------------------------------------------------------------ 7 costs
    A(r"\section{交易成本与可实施性}\label{sec:cost}")
    A(f"费后指数按当期实际制度扣费：佣金随历史下调、印花税按各时期税率（2008 年 9 月起仅卖出征收、2023 年 8 月起减半）、过户费与固定冲击成本。费前年化 {pct(cost['Gross_CAGR'])}，费后 {pct(cost['Net_CAGR'])}，"
      f"年化成本拖累 {pct(cost['Annual_Cost_Drag'])}；{years:.0f} 年累计支付的成本相当于期末净值的 {pct(cost['Total_Cost_Paid_fraction_of_NAV'], 1)}。各年单边换手见表 \\ref{{tab:turnover}}。"
      f"成分股全部为沪深300成分、流动性充足，半年一次调仓、单边换手约 {pct(g('Average_Rebalance_Turnover', S), 0)}，对资金规模不敏感；主要摩擦来自停牌股票（回测按停牌前价格成交）以及涨跌停板日无法足量成交。")
    tb = ctx["turnover_by_year"].copy(); half = (len(tb) + 1) // 2
    pad = lambda lst, n: list(lst) + [""] * (n - len(lst))  # noqa: E731
    tb_w = pd.DataFrame({"年份": tb["year"].iloc[:half].astype(int).astype(str).values, "单边换手": tb["one_way_turnover"].iloc[:half].map(lambda v: pct(v, 0)).values,
                         "年份 ": pad(tb["year"].iloc[half:].astype(int).astype(str).values, half), "单边换手 ": pad(tb["one_way_turnover"].iloc[half:].map(lambda v: pct(v, 0)).values, half)})
    A(table(tb_w, "各日历年的单边换手率", "tab:turnover", colspec=r"@{}lrlr@{}", escape=False))

    # ------------------------------------------------------------------ 8 HSMO
    A(r"\section{与 HSI300-Momentum-Strategy 的对比}\label{sec:hsmo}")
    A(r"GitHub 项目 HSI300-Momentum-Strategy \cite{hsmo} 同样受 SPMO 启发在沪深300内做动量选股，但设计取向不同：它是一个\textbf{集中持股的量化组合}（Top-20 等权、季度调仓、行业上限），"
      r"而本文是一只\textbf{指数}（约 50 只、市值乘以分数加权、半年调仓、缓冲与上限）。本节先比较规则（表 \ref{tab:hsmo_rules}），再用本文的数据复现其规则，在相同窗口和相同数据上比较结果。")
    dif = pd.DataFrame([
        ["数据源", "BaoStock：每月末 hs300 成分快照、后复权收盘价、当前行业分类", "中证公告逐日点位成分（2005 起）、新浪 / 东财原始价格自建复权、季度财报与公告日"],
        ["成分股口径", "调仓日采用最近一个月末快照", "调仓日当日有效成分；调仓期间被剔除的持仓同日剔除"],
        ["盈利筛选", "无", "S\\&P Financial Viability"],
        ["动量信号", "$P_{t-22}/P_{t-253}-1$ 除以 252 日日收益标准差（窗口含最近一月），$T-1$ 计算 $T$ 交易", "12--1 月价格变动除以同窗口日收益标准差；参考日计算，三周后实施"],
        ["标准化 / 分数", "无（按原始风险调整动量排序）", "横截面 $Z$ 分数（$\\pm 3$ 截断）→ Momentum Score"],
        ["选样", "Top 20，每个（当前）行业最多 5 只", "合格股票前 20\\%（约 45--55 只），S\\&P 20\\% 缓冲，母指数资格优先"],
        ["权重", "等权", f"自由流通市值 $\\times$ Score，公司上限 $\\min({cap_abs*100:.0f}\\%,\\,{cap_mult:g}\\,w^P)$"],
        ["调仓频率", "季度（自然季末最后一个交易日收盘）", "半年（2 / 8 月末参考，3 / 9 月第三个星期五实施）"],
        ["收益口径", "策略用后复权价（含股息）对沪深300价格指数", "价格指数与全收益指数分别对应比较"],
        ["成本", "成交金额的 15\\,bp（买卖同费率）", "按历史制度的佣金、印花税、过户费与冲击成本"],
        ["回测区间", "2019-01-01 → 2025-12-20（7 年）", f"{start} → {end}（{years:.0f} 年）"],
        ["资格审计", "无", "每期自动审计成员资格、财报公告日、动量数据可得性"],
    ], columns=["项目", "HSI300-Momentum-Strategy", "本文"])
    A(table(dif, "两个项目的规则差异", "tab:hsmo_rules", colspec=r"@{}p{1.9cm}p{6.2cm}p{6.6cm}@{}", escape=False, header_escape=False, size=r"\footnotesize"))
    A(r"\subsection{其公布结果与在本文数据上的复现（2019--2025）}")
    A(r"我们按其代码逐条复现规则（信号、季末调仓、等权、漂移、15\,bp 成本），只把数据换成本文的点位成分与复权价格。行业上限使用当前中证一级行业分类（其代码同样使用当前分类），"
      r"历史上已不在沪深300的股票无分类，归入“未知”一类同样受 5 只上限约束。表 \ref{tab:hsmo_win} 与表 \ref{tab:hsmo_ann} 给出结果；其公布值转录自该仓库 README 中的结果图。")
    A(table(stats_df(win_rows), f"HSMO 仓库公布值与在本文数据上的复现，{w0} 至 {w1}；夏普比率按其代码以 2\\% 无风险利率计算", "tab:hsmo_win", colspec=r"@{}lrrrrr@{}", escape=False, header_escape=False, size=r"\footnotesize"))
    ann_rep = HSMO_REPORTED["annual"]
    yrs = list(ann_rep.keys())
    ann_cmp = pd.DataFrame({"年份": [str(y) for y in yrs], "公布：策略": [pct(ann_rep[y][0], 1) for y in yrs], "公布：沪深300": [pct(ann_rep[y][1], 1) for y in yrs],
                            "复现：行业上限": [pct(_ann(hs['win_cap']['levels']).get(y, np.nan), 1) for y in yrs], "复现：无上限": [pct(_ann(hs['win_nocap']['levels']).get(y, np.nan), 1) for y in yrs],
                            "本指数 TR 费后": [pct(_ann(hw[STN]).get(y, np.nan), 1) for y in yrs], "沪深300（本文数据）": [pct(_ann(hw[B]).get(y, np.nan), 1) for y in yrs]})
    A(table(ann_cmp, "逐年收益：HSMO 公布值、复现值与本指数", "tab:hsmo_ann", colspec=r"@{}lrrrrrr@{}", escape=False, size=r"\footnotesize", fit=True))
    A(f"复现结果与其公布值方向一致、量级接近：沪深300年度收益逐年完全相同（两边都用官方价格指数），策略年化 {pct(cmp_win['复现：HSMO 规则，行业上限（当前分类）']['CAGR'], 1)} 对公布的 {pct(rep_sum['年化收益'][0], 1)}，"
      f"差异主要来自 2020 年（复现 {pct(_ann(hs['win_cap']['levels']).get(2020, np.nan), 1)} 对公布 {pct(ann_rep[2020][0], 1)}）和 2022 年，源于成分股快照口径（月末快照对逐日）、复权价格与行业分类的差别，而非规则本身。"
      f"因此可以把两条曲线放在同一数据上比较：2019--2025 年 HSMO 规则 \\textyen 100 → {yen(hsmo_final['win_nocap'], 0)}（无行业上限）/ {yen(hsmo_final['win_cap'], 0)}（行业上限），本指数全收益费后 {yen(ours_win, 0)}，沪深300全收益 {yen(bt_win, 0)}。"
      f"HSMO 规则在这 7 年里收益更高、最大回撤更小（{pct(hsmo_dd_win['max_drawdown'], 1)} 对本指数全收益 {pct(M.max_drawdown(hw[STN])['max_drawdown'], 1)}）——差别主要在 2021--2022 年：季度调仓的等权组合在 2021 年 2 月核心资产见顶后一个季度内就转向，而半年调仓的本指数持有 2 月底确定的名单直到 9 月。")
    A(r"\subsection{全样本、同数据比较（2005--2026）}")
    A(table(stats_df(full_rows), f"{start} 至 {end}，两套规则在同一数据上的表现", "tab:hsmo_full", colspec=r"@{}lrrrrr@{}", escape=False, header_escape=False, size=r"\footnotesize"))
    k_nc, k_pc, k_cap, k_ours = K_NC, K_PC, K_CAP, K_OURS
    A(f"把窗口拉长到 {years:.0f} 年（表 \\ref{{tab:hsmo_full}}、图 \\ref{{fig:hsmo}}、图 \\ref{{fig:dd3}}），HSMO 规则（无行业上限）\\textyen 100 → {yen(hsmo_final['full_nocap'], 0)}，本指数全收益费后 {yen(float(lv[STN].iloc[-1]), 0)}，沪深300全收益 {yen(g('Final_Value_of_100', BT), 0)}。"
      f"HSMO 规则的年化收益仍高出本指数约 {(cmp_full[k_nc]['CAGR'] - cmp_full[k_ours]['CAGR']) * 100:.1f} 个百分点，但代价与限定条件如下。")
    A(r"\begin{itemize}")
    A(f"\\item \\textbf{{集中度。}}20 只等权对约 50 只，有效成分数 20 对 {hc['effective_n'].mean():.0f}；年化波动 {pct(cmp_full[k_nc]['Volatility'], 1)} 对 {pct(cmp_full[k_ours]['Volatility'], 1)}。")
    A(f"\\item \\textbf{{换手。}}季度调仓、无缓冲，双边换手平均每季 {pct(hs_to.mean(), 0)}，折合单边约 {pct(hsmo_turnover_year, 0)}/年，是本指数（{pct(g('Average_Annual_Turnover', S), 0)}/年）的 {hsmo_turnover_year / g('Average_Annual_Turnover', S):.1f} 倍；"
      f"换用本文的历史成本模型后其年化降至 {pct(cmp_full[k_pc]['CAGR'], 1)}（2019--2025 年窗口 {pct(cmp_win['复现：HSMO 规则，无上限，本项目成本模型']['CAGR'], 1)}）。")
    A(f"\\item \\textbf{{行业中性规则不稳定。}}按当前行业分类施加“每行业 5 只”在全样本上把年化从 {pct(cmp_full[k_nc]['CAGR'], 1)} 推高到 {pct(cmp_full[k_cap]['CAGR'], 1)}，"
      f"其中 2006 年一年相差 {(_ann(hs['full_cap']['levels']).get(2006, np.nan) - _ann(hs['full_nocap']['levels']).get(2006, np.nan)) * 100:+.0f} 个百分点——差异来自把早年已退市或已调出的股票统统归入“未知”行业并受同一上限约束，是\\emph{{当前分类倒推历史}}带来的伪影，不能视为该规则的真实增益。".replace("+", "$+$"))
    A(f"\\item \\textbf{{回撤。}}2008 年两者都亏损约 70\\%（HSMO 规则 {pct(hsmo_dd_full['max_drawdown'], 1)}，{hsmo_dd_full['peak_date'].date()} → {hsmo_dd_full['bottom_date'].date()}），"
      f"HSMO 于 {hsmo_dd_full['recovery_date'].date() if pd.notna(hsmo_dd_full['recovery_date']) else 'n/a'} 收复，本指数全收益于 {su.loc['MDD_Recovery_Date', ST]} 收复。".replace("→", "$\\rightarrow$"))
    A(f"\\item \\textbf{{样本期。}}该项目公布的 2019--2025 年恰好避开了 2008--2016 年动量策略在 A 股长期跑输的阶段（第 \\ref{{sec:results}} 节），7 年样本的年化超额（{pct(rep_sum['年化收益'][0] - rep_sum['年化收益'][1], 1)}）不应外推为长期预期。")
    A(r"\item \textbf{口径。}其策略曲线含股息（后复权）而基准为价格指数，7 年内约多计 2 个百分点/年的股息；本文分别给出价格与全收益口径。")
    A(r"\end{itemize}")
    A(figure(rel(fig["hsmo"]), "HSMO 规则（在本文数据上复现）与本指数：其 2019--2025 年窗口（左）与 2005--2026 年全样本（右），\\textyen 100，对数坐标", "fig:hsmo"))
    A(figure(rel(fig["dd"]), "回撤比较：沪深300、本指数（价格）、HSMO 规则（全收益、费后）", "fig:dd3"))
    A(r"\subsection{评述}")
    A(r"两种设计代表了动量因子的两种用法。HSI300-Momentum-Strategy 是\textbf{高活跃度的因子组合}：更集中、更快的调仓让它在趋势切换后能更早转向，历史收益更高，但波动、换手、成本和单一行业与个股风险也更高，"
      r"且其点位数据（月末快照、当前行业分类）与成本假设（15\,bp）在 2005--2008 年的制度环境下偏乐观。本文是\textbf{可指数化的规则}：更宽的成分、市值倾斜的权重、缓冲和上限带来更低的换手和更接近母指数的风险特征（$\beta\approx 1$、跟踪误差约 10\%），"
      r"代价是对趋势反转反应更慢（2021 年、2024 年 9 月）。两者在同一数据上都长期战胜沪深300，也都在 2008、2011、2015--2016 年经历长时间跑输——\textbf{动量在 A 股大盘股中的有效性是周期性的}，任何以 2019 年以后样本为依据的收益预期都需要打折。")


    # ------------------------------------------------------------------ 9 robustness / design space
    if ctx.get("design"):
        d = ctx["design"]; ds = d["summary"]; dg = d["grid"]; dp = d["phase"]; dd_ = d["dispersion"]
        A(r"\section{设计空间的稳健性与“折中”方案}\label{sec:robust}")
        A(f"第 \\ref{{sec:hsmo}} 节的两套规则在多个维度上同时不同。本节把这些维度\\textbf{{逐项}}拆开：从本指数出发，每次只改一个设计选择，"
          f"其余全部保持不变（同一套点位数据、同一指数引擎、同一历史成本模型、同一公共区间、全收益费后口径），最后回答“中间地带是否存在更好的设计”。"
          f"\\textbf{{本节是诊断性分析：已发布指数的规则不因此改变，配置文件中的参数也没有按结果调整。}}")
        A(r"\subsection{逐维度敏感性}")
        gt = pd.DataFrame({"维度": dg["group"].map(esc), "变体": dg["variant"].map(esc), "年化（费后）": dg["CAGR_net"].map(lambda v: pct(v, 1)),
                           "波动": dg["Vol"].map(lambda v: pct(v, 1)), "最大回撤": dg["MaxDD"].map(lambda v: pct(v, 1)),
                           "单边换手/年": dg["Turnover_pa"].map(lambda v: pct(v, 0)), "成分数": dg["N"].map(lambda v: f"{v:.0f}"),
                           "有效成分数": dg["EffN"].map(lambda v: f"{v:.0f}"),
                           "前半段": dg["CAGR_net_first_half"].map(lambda v: pct(v, 1)), "后半段": dg["CAGR_net_second_half"].map(lambda v: pct(v, 1))})
        A(table(gt, "设计维度逐项变动（全收益、费后；前半段 = 2016 年之前，后半段 = 2016 年之后）", "tab:design",
                colspec=r"@{}llrrrrrrrr@{}", escape=False, size=r"\footnotesize", fit=True))
        base_c = float(dg["CAGR_net"].iloc[0])
        A(f"全部 {len(dg)} 个变体的费后年化收益落在 {pct(ds['design_min'], 1)}--{pct(ds['design_max'], 1)} 之间，基准（本指数）为 {pct(base_c, 1)}。"
          f"可以直接读出四条结论：(i) \\textbf{{更集中并不更好}}——把成分数压到 10\\%、5\\% 或固定 20 只，收益不升反降，而波动、回撤和换手全部上升；"
          f"(ii) \\textbf{{加权方案几乎不影响长期收益}}——等权、纯市值、$\\sqrt{{\\FMC}}\\times$Score 与本指数相差不到 0.1 个百分点，"
          f"区别只在前后两段的分布（等权前半段更强、后半段更弱）；(iii) \\textbf{{财务资格筛选值得保留}}——去掉后年化下降 "
          f"{100 * (base_c - float(dg.loc[dg['variant'].str.startswith('E'), 'CAGR_net'].iloc[0])):.2f} 个百分点，且前后两段一致；"
          f"(iv) \\textbf{{把三周的实施延迟去掉、改在参考日收盘交易，单独看是有害的}}（{pct(float(dg.loc[dg['variant'].str.startswith('L'), 'CAGR_net'].iloc[0]), 1)}），"
          f"只有与季度调仓、高集中度组合在一起时才转为有利——这本身就说明这些差异不是稳定的结构性效应。")
        A(r"\subsection{同一规则、不同调仓月份：设计差异小于日历噪声}")
        A(f"上述差异是否值得据以改变设计？一个直接的检验是：\\textbf{{保持规则完全不变，只把参考月份整体平移}}。本指数用 2 月 / 8 月，"
          f"这一选择来自 S\\&P 的日程惯例，没有任何经济含义。把它换成其余 5 种半年相位，同样的规则给出的费后年化收益是 "
          f"{pct(ds['phase_min'], 1)}--{pct(ds['phase_max'], 1)}（标准差 {100 * ds['phase_std']:.1f} 个百分点，极差 "
          f"{100 * (ds['phase_max'] - ds['phase_min']):.1f} 个百分点），\\textbf{{比上一小节全部设计变体之间的差距还要大}}（图 \\ref{{fig:design}} 左）。"
          f"换言之，在 21 年样本上，1--2 个百分点的年化差异不足以区分两种设计的优劣。")
        A(f"这一结果同样适用于本文自己的指数：已发布的 2 月 / 8 月相位年化 {pct(ds['published_cagr'], 1)}，而六个相位的平均为 {pct(ds['phase_mean'], 1)}；"
          f"相对沪深300全收益（{pct(ds['bench_cagr'], 1)}）的超额由 {pct(ds['published_cagr'] - ds['bench_cagr'], 1, True)} 降到 "
          f"{pct(ds['phase_mean'] - ds['bench_cagr'], 1, True)}。也就是说，\\textbf{{已报告超额收益中约 "
          f"{100 * (ds['published_cagr'] - ds['phase_mean']):.1f} 个百分点来自调仓月份的运气}}，而非规则本身。这一点在第 \\ref{{sec:results}} 节的结论中必须一并考虑。")
        A(r"\subsection{可以稳健改善的一件事：分批调仓}")
        dt = pd.DataFrame({"分批数": dd_["分批数"].astype(int).astype(str), "可能组合数": dd_["组合数"].astype(int).astype(str),
                           "年化均值": dd_["CAGR 均值"].map(lambda v: pct(v, 1)), "最低": dd_["CAGR 最低"].map(lambda v: pct(v, 1)),
                           "最高": dd_["CAGR 最高"].map(lambda v: pct(v, 1)), "极差（pp）": dd_["极差(pp)"].map(lambda v: f"{v:.1f}"),
                           "标准差（pp）": dd_["标准差(pp)"].map(lambda v: f"{v:.2f}"), "波动": dd_["波动"].map(lambda v: pct(v, 1)),
                           "Sharpe": dd_["Sharpe"].map(lambda v: f"{v:.3f}"), "最大回撤": dd_["最大回撤"].map(lambda v: pct(v, 1))})
        A(table(dt, "把同一套规则拆成 $k$ 个错开调仓月份的子组合（每个子组合占 $1/k$ 资金，单位资金的换手率不变）", "tab:tranche",
                colspec=r"@{}llrrrrrrrr@{}", escape=False, size=r"\footnotesize", fit=True))
        A(f"既然“哪个月调仓”是纯粹的运气来源，就应当把它\\textbf{{分散掉}}，而不是去挑一个幸运的月份：把资金分成 $k$ 份、各自遵循同样的规则但错开调仓月份"
          f"（动量文献中的重叠组合，\\citealp{{jegadeesh1993}}）。这是唯一一个\\emph{{事前}}就能论证、无需回测支持的改进：单位资金的换手率不变，"
          f"因而成本不变，但结果对日历选择的依赖被消除。$k=2$ 已经把极差从 {dd_['极差(pp)'].iloc[0]:.1f} 个百分点压到 {dd_['极差(pp)'].iloc[1]:.1f}，"
          f"$k=3$ 压到 {dd_['极差(pp)'].iloc[2]:.1f}，$k=6$（逐月错开）为零（图 \\ref{{fig:design}} 右）；同时平均波动由 {pct(float(dd_['波动'].iloc[0]), 1)} 降至 "
          f"{pct(float(dd_['波动'].iloc[3]), 1)}，平均 Sharpe 由 {float(dd_['Sharpe'].iloc[0]):.3f} 升至 {float(dd_['Sharpe'].iloc[3]):.3f}，"
          f"平均最大回撤由 {pct(float(dd_['最大回撤'].iloc[0]), 1)} 改善到 {pct(float(dd_['最大回撤'].iloc[3]), 1)}。分批后的组合年化 {pct(ds['tranched_cagr'], 1)}，"
          f"相对沪深300全收益超额 {pct(ds['tranched_cagr'] - ds['bench_cagr'], 1, True)}——这是对该规则超额收益更诚实的估计。")
        A(figure(rel(fig["design"]), "左：逐维度设计变体的费后年化收益，灰带为同一规则在 6 种调仓相位下的区间；右：分批数与结果离散度", "fig:design"))
        A(r"\subsection{这些差异有多少是真的？自助法区间}")
        sg = d["significance"]
        st_ = pd.DataFrame({"比较": sg["比较"].map(esc), "年化差（pp）": sg["年化差(pp)"].map(lambda v: f"{v:+.2f}".replace("-", "$-$")),
                            "95\% 区间（pp）": [f"[{a:+.2f}, {b:+.2f}]".replace("-", "$-$") for a, b in zip(sg["区间下(pp)"], sg["区间上(pp)"])],
                            "$t$（Newey--West）": sg["t(NW)"].map(lambda v: f"{v:.2f}".replace("-", "$-$")),
                            "$P(\Delta>0)$": sg["P(差>0)"].map(lambda v: f"{v:.2f}"), "月数": sg["月数"].astype(int).astype(str)})
        A(table(st_, "月度收益差的移动分块自助法（分块 12 个月，2000 次重抽样）与 Newey--West $t$ 统计量", "tab:sig",
                colspec=r"@{}lrrrrr@{}", escape=False, header_escape=False, size=r"\footnotesize", fit=True))
        A(f"跟踪误差 {pct(ds['te'], 1)}、样本长度 {M.years_between(lv.index[0], lv.index[-1]):.0f} 年，意味着年化超额收益的标准误约 "
          f"{100 * ds['se_excess']:.1f} 个百分点。表 \\ref{{tab:sig}} 证实了这一点：\\textbf{{所有比较的 95\\% 区间都跨过零}}——"
          f"包括本指数相对沪深300全收益本身（{sg['年化差(pp)'].iloc[0]:+.2f} pp，区间 [{sg['区间下(pp)'].iloc[0]:+.2f}, {sg['区间上(pp)'].iloc[0]:+.2f}]，"
          f"$t={sg['t(NW)'].iloc[0]:.2f}$）。因此“哪一种设计更好”在本样本上不是一个可以回答的问题；只有那些\\emph{{不依赖收益差}}的性质"
          f"（换手、集中度、回撤、结果离散度）才是可以判断的。区间最窄的两项是去掉财务筛选（{sg['区间下(pp)'].iloc[6]:+.2f} 到 {sg['区间上(pp)'].iloc[6]:+.2f} pp，"
          f"$P(\\text{{差}}>0)={sg['P(差>0)'].iloc[6]:.2f}$，即九成的重抽样认为去掉它有害）和波动率目标（见下）。".replace("%", r"\%").replace(r"\\%", r"\%"))
        A(r"\subsection{还有没有更好的？波动率目标}")
        ov = d["overlay"]
        ot = pd.DataFrame({"目标波动": ov["目标波动"].map(esc), "估计窗口": ov["估计窗口"].map(lambda v: "—" if pd.isna(v) else f"{v:.0f} 日"),
                           "年化": ov["CAGR"].map(lambda v: pct(v, 2)), "波动": ov["波动"].map(lambda v: pct(v, 1)),
                           "Sharpe": ov["Sharpe"].map(lambda v: f"{v:.3f}"), "最大回撤": ov["最大回撤"].map(lambda v: pct(v, 1)),
                           "平均仓位": ov["平均仓位"].map(lambda v: pct(v, 0)), "叠加换手/年": ov["叠加换手/年"].map(lambda v: pct(v, 0))})
        A(table(ot, "波动率目标叠加：仓位 $=\\min(1,\\ \\sigma^{\\ast}/\\hat\\sigma_{t-1})$，空仓部分按 2\\% 计息，仓位变动按 15\\,bp 计费；"
                     f"全部自 {esc(ds['burn_start'])} 起（扣除 2 年估计期）", "tab:overlay", colspec=r"@{}llrrrrrr@{}", escape=False,
                header_escape=False, size=r"\footnotesize", fit=True))
        base = ov.iloc[0]
        best = ov.iloc[1:].loc[ov.iloc[1:]["Sharpe"].idxmax()]
        exp126 = ov[(ov["目标波动"] == "扩展窗口（无前视）") & (ov["估计窗口"] == 126)].iloc[0]
        nb = int((ov.iloc[1:]["Sharpe"] > base["Sharpe"]).sum())
        A(f"分批解决的是运气，不是风险。文献中对动量策略最稳健的一项改造是\\textbf{{按自身波动率调节仓位}}"
          f"（\\citealp{{barroso2015}}；\\citealp{{daniel2016}}）：动量组合的波动率高度可预测，而其崩塌集中在高波动期。"
          f"把这一叠加用在本指数上（仓位上限 100\\%，即只降杠杆不加杠杆，空仓部分计息，仓位变动计费）：{len(ov) - 1} 组参数中有 {nb} 组的 Sharpe "
          f"同时优于基准，最大回撤在\\emph{{全部}} {len(ov) - 1} 组中都从 {pct(base['最大回撤'], 1)} 改善到 {pct(ov.iloc[1:]['最大回撤'].max(), 1)} 以内。"
          f"其中唯一完全不含前视的设定（目标 = 迄今为止实现波动的扩展窗口估计，126 日估计窗口）年化 {pct(exp126['CAGR'], 2)}（基准 {pct(base['CAGR'], 2)}），"
          f"波动 {pct(exp126['波动'], 1)}（基准 {pct(base['波动'], 1)}），Sharpe {exp126['Sharpe']:.3f}（基准 {base['Sharpe']:.3f}），"
          f"最大回撤 {pct(exp126['最大回撤'], 1)}（基准 {pct(base['最大回撤'], 1)}），平均仓位 {pct(exp126['平均仓位'], 0)}，叠加换手 {pct(exp126['叠加换手/年'], 0)}/年。"
          f"收益上的改进依然不显著（表 \\ref{{tab:sig}} 末行，$t={sg['t(NW)'].iloc[-1]:.2f}$），但\\textbf{{风险上的改进是机械的}}："
          f"它不依赖动量是否继续有效，只依赖波动率的可预测性——后者在所有 {len(ov) - 1} 组参数上都成立（图 \\ref{{fig:evidence}}）。")
        A(figure(rel(fig["evidence"]), "左：各项比较的自助法 95\\% 区间；右：波动率目标叠加前后的回撤", "fig:evidence"))
        A(r"这项改造的代价必须一并说明：它把指数变成一只\emph{仓位可变}的产品（需要现金账户，且不再是一条可被跟踪的纯指数），"
          r"叠加换手每年 11\%--95\%（视估计窗口而定），在急速反弹的初期会因仓位偏低而落后，且以上结论只覆盖 2008 年以后的样本。")

        A(r"\subsection{结论：折中方案是什么}")
        A(r"把上面的证据合起来，两套规则之间\textbf{沿“更集中 / 更高频”方向的中间点没有可靠的优势}——那一维度上的差异在日历噪声之内，"
          r"而代价（波动、回撤、换手、容量）是确定的。真正值得采纳的折中是：")
        A(r"\begin{enumerate}[leftmargin=2em,itemsep=2pt]")
        A(f"\\item \\textbf{{保留本指数的宽度、财务资格筛选、缓冲与权重上限。}}这些选择要么带来稳定的收益（筛选 "
          f"{100 * (base_c - float(dg.loc[dg['variant'].str.startswith('E'), 'CAGR_net'].iloc[0])):.2f} 个百分点），要么在收益相同的情况下降低波动、回撤与换手（宽度）。")
        A(r"\item \textbf{把调仓时点分成 2--3 批。}换手率与成本不变，消除掉一半到三分之二的日历运气，平均 Sharpe 与最大回撤同时小幅改善。"
          r"理由是事前的（分散一个无信息的选择），不依赖样本内表现。")
        A(f"\\item \\textbf{{若产品允许仓位可变，波动率目标是效果最大的一项改造。}}最大回撤由 {pct(float(d['overlay']['最大回撤'].iloc[0]), 1)} 改善到 "
          f"{pct(float(d['overlay']['最大回撤'].iloc[2]), 1)}（无前视设定），9 组参数中 9 组的 Sharpe 与回撤同时改善；收益上的改进不显著，"
          f"但风险上的改进不依赖动量是否继续有效。代价是它不再是一条可跟踪的纯指数。")
        A(r"\item \textbf{不要为 1--2 个百分点的样本内年化收益去调整成分数、频率或加权方案。}本节的证据表明这个量级低于噪声；"
          r"按它调参得到的将是拟合结果，而不是改进。")
        A(r"\end{enumerate}")
        A(r"需要说明的是，分批实施改变的是\emph{组合的运作方式}而非指数定义：作为一只公开指数，本文的 CSI300\_SP\_FV\_SPMO 仍按 S\&P 的"
          r"半年、提前公告的日程编制；分批属于跟踪该规则的\emph{产品}层面的实现选择。")

    # ------------------------------------------------------------------ 9 conclusion & 17 answers
    A(r"\section{结论}\label{sec:conclusion}")
    A(f"把 S\\&P 500 Momentum 的规则平移到沪深300并叠加 Financial Viability 盈利资格，在 2005--2026 年得到一条\\textbf{{长期期望为正、但需要忍受多年跑输}}的规则：年化收益由 {pct(cagr_b)} 提高到 {pct(cagr_s)}（费后 {pct(cagr_sn)}），"
      f"Sharpe 由 {g('Sharpe_Ratio', B):.2f} 升至 {g('Sharpe_Ratio', S):.2f}，但波动与最大回撤更大，超额收益集中在 2017--2020 年，月度超额在统计上不显著。超额几乎全部来自选股而非 SPMO 式加权；"
      f"本文的自由流通代理对国有大盘股的高配使报告的业绩偏保守。相同数据上，更集中、更高频的 Top-20 等权季度动量收益更高，但换手约两倍、集中度更高，其行业中性规则依赖当前分类而在历史上不可靠。"
      + (f"第 \\ref{{sec:robust}} 节进一步表明，两种设计之间的差距小于“在哪个月调仓”这一无信息选择造成的差异（同一规则、6 个相位：{pct(ctx['design']['summary']['phase_min'], 1)}--{pct(ctx['design']['summary']['phase_max'], 1)}）；"
         f"已报告超额中约 {100 * (ctx['design']['summary']['published_cagr'] - ctx['design']['summary']['phase_mean']):.1f} 个百分点应归于日历运气。"
         f"所有比较的自助法 95\\% 区间都跨过零（包括本指数相对沪深300本身），因此“哪种设计更好”在本样本上不可判定；能够判断的只有不依赖收益差的性质："
         f"把调仓时点分成 2--3 批可零成本消除日历运气，按自身波动率调节仓位可把最大回撤由 {pct(float(ctx['design']['overlay']['最大回撤'].iloc[0]), 0)} 降到 "
         f"{pct(float(ctx['design']['overlay']['最大回撤'].iloc[2]), 0)}（9 组参数一致），代价是不再是纯指数。" if ctx.get("design") else "")
      + f"研究任务书要求回答的 17 个问题逐条如下。")
    ann_s = ann.set_index("Date"); strong = ann_s[S].nlargest(3); weak = ann_s[S].nsmallest(3); bestex = ann_s["Excess"].nlargest(3); worstex = ann_s["Excess"].nsmallest(3)
    jj = lambda ser: "、".join(f"{y} 年（{pct(v, 1, True)}）" for y, v in ser.items())  # noqa: E731
    ans = [
        ("回测起止日期", f"{start} 至 {end}（首个参考日 {rb['reference_date'].iloc[0].date()}，首次实施 {rb['implementation_date'].iloc[0].date()}；{n_years} 个日历年，含首尾不完整年份）。"),
        ("\\textyen 100 投资沪深300", f"{yen(g('Final_Value_of_100', B))}（价格指数）；含股息 {yen(g('Final_Value_of_100', BT))}。"),
        ("\\textyen 100 投资策略", f"{yen(g('Final_Value_of_100', S))}（费前）/ {yen(g('Final_Value_of_100', SN))}（费后）；全收益费前 {yen(g('Final_Value_of_100', ST))}。"),
        ("两者 CAGR", f"沪深300 {pct(cagr_b)}，策略 {pct(cagr_s)}（全收益口径 {pct(g('CAGR', BT))} 对 {pct(g('CAGR', ST))}）。"),
        ("策略年化超额收益", f"{pct(g('Annualized_Excess_Return', S))}（费后 {pct(g('Annualized_Excess_Return', SN))}）；Jensen $\\alpha$ {pct(g('Alpha_vs_CSI300', S))}，$\\beta$ {g('Beta_vs_CSI300', S):.2f}。"),
        ("两者最大回撤", f"沪深300 {pct(g('Maximum_Drawdown', B))}（{su.loc['MDD_Peak_Date', B]} → {su.loc['MDD_Bottom_Date', B]}）；策略 {pct(g('Maximum_Drawdown', S))}（{su.loc['MDD_Peak_Date', S]} → {su.loc['MDD_Bottom_Date', S]}）。".replace("→", "$\\rightarrow$")),
        ("两者 Sharpe", f"沪深300 {g('Sharpe_Ratio', B):.2f}，策略 {g('Sharpe_Ratio', S):.2f}（费后 {g('Sharpe_Ratio', SN):.2f}）。"),
        ("两者年化波动率", f"沪深300 {pct(g('Annualized_Volatility', B))}，策略 {pct(g('Annualized_Volatility', S))}。"),
        ("策略平均换手率", f"单边每次调仓 {pct(g('Average_Rebalance_Turnover', S), 1)}，每年 {pct(g('Average_Annual_Turnover', S), 1)}（最高单次 {pct(g('Maximum_Rebalance_Turnover', S), 1)}）。"),
        ("扣除交易成本后的策略 CAGR", f"{pct(cost['Net_CAGR'])}（成本拖累 {pct(cost['Annual_Cost_Drag'])}/年）。"),
        ("策略最强的年份", f"{jj(strong)}；超额最大：{jj(bestex)}。"),
        ("策略最差的年份", f"{jj(weak)}；超额最差：{jj(worstex)}。"),
        ("最大 Momentum Crash", f"{c0['Start_Date']} 至 {c0['End_Date']}，策略 {pct(c0['Strategy_3M_Return'], 1, True)} 对沪深300 {pct(c0['CSI300_3M_Return'], 1, True)}（相对 {pct(c0['Relative_3M'], 1, True)}）。"),
        ("是否长期稳定跑赢沪深300", f"长期跑赢但不稳定：{n_years} 年中 {wins} 年跑赢，2005--2016 年累计跑输且相对强度回撤 {pct(rs_dd['max_drawdown'], 1)}、历时 {rs_dd['days_to_bottom']} 天，超额几乎全部来自 2017 年以后；月度超额的 $t$ 统计量 {mon['excess_t_stat']:.2f}，统计上不显著。"),
        ("3 年和 5 年滚动超额收益胜率", f"3 年 {pct(roll['rolling_3y_cagr_excess_win_rate'], 1)}，5 年 {pct(roll['rolling_5y_cagr_excess_win_rate'], 1)}（1 年 {pct(roll['rolling_1y_return_excess_win_rate'], 1)}）。"),
        ("相对强度长期趋势", f"由 1.00 升至 {rs_final:.2f}（年化 {pct(rs_final ** (1 / years) - 1, 2, True)}），但先经历 2007--2016 年的长期下行（{pct(rs_dd['max_drawdown'], 1)}），2017 年后转为上行并于 2019 年收复。"),
        ("Financial Viability + SPMO 是否显著改善沪深300的长期复利与风险调整收益",
         f"长期复利有实质改善（年化 {pct(cagr_b)} → {pct(cagr_s)}，费后 {pct(cagr_sn)}；{years:.0f} 年终值高 {(g('Final_Value_of_100', S) / g('Final_Value_of_100', B) - 1) * 100:.0f}\\%），"
         f"风险调整收益温和改善（Sharpe {g('Sharpe_Ratio', B):.2f} → {g('Sharpe_Ratio', S):.2f}，Sortino {g('Sortino_Ratio', B):.2f} → {g('Sortino_Ratio', S):.2f}，信息比率 {g('Information_Ratio', S):.2f}），"
         f"但波动和最大回撤更大、超额收益集中在少数年份且统计上不显著；把调仓月份的相位差异计入后，超额的中枢估计还要再降约 "
         f"{100 * (ctx['design']['summary']['published_cagr'] - ctx['design']['summary']['phase_mean']):.1f} 个百分点（第 \\ref{{sec:robust}} 节）。"
         f"结论：\\textbf{{有改善，但不能称为“显著”}}。".replace("→", "$\\rightarrow$") if ctx.get("design") else
         f"但波动和最大回撤更大、超额收益集中在少数年份且统计上不显著。结论：\\textbf{{有改善，但不能称为“显著”}}。".replace("→", "$\\rightarrow$")),
    ]
    A(r"\begin{enumerate}[leftmargin=2em,itemsep=2pt]")
    for q, a_ in ans:
        A(f"\\item \\textbf{{{q}}}\\quad {a_}")
    A(r"\end{enumerate}")

    # ------------------------------------------------------------------ references
    A(r"\begin{thebibliography}{99}\footnotesize")
    A(r"\bibitem[Jegadeesh and Titman(1993)]{jegadeesh1993} Jegadeesh, N., and S. Titman. 1993. Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency. \emph{Journal of Finance} 48(1): 65--91.")
    A(r"\bibitem[Carhart(1997)]{carhart1997} Carhart, M. M. 1997. On Persistence in Mutual Fund Performance. \emph{Journal of Finance} 52(1): 57--82.")
    A(r"\bibitem[Asness et al.(2013)]{asness2013} Asness, C. S., T. J. Moskowitz, and L. H. Pedersen. 2013. Value and Momentum Everywhere. \emph{Journal of Finance} 68(3): 929--985.")
    A(r"\bibitem[Daniel and Moskowitz(2016)]{daniel2016} Daniel, K., and T. J. Moskowitz. 2016. Momentum Crashes. \emph{Journal of Financial Economics} 122(2): 221--247.")
    A(r"\bibitem[Barroso and Santa-Clara(2015)]{barroso2015} Barroso, P., and P. Santa-Clara. 2015. Momentum Has Its Moments. \emph{Journal of Financial Economics} 116(1): 111--120.")
    A(r"\bibitem[S\&P Dow Jones Indices(2026a)]{spdji_momentum} S\&P Dow Jones Indices. 2026. \emph{S\&P Momentum Indices Methodology}. New York: S\&P Global.")
    A(r"\bibitem[S\&P Dow Jones Indices(2026b)]{spdji_us} S\&P Dow Jones Indices. 2026. \emph{S\&P U.S. Indices Methodology}. New York: S\&P Global.")
    A(r"\bibitem[S\&P Dow Jones Indices(2015)]{spdji_bovespa} S\&P Dow Jones Indices. 2015. \emph{S\&P/BOVESPA Momentum Index Methodology}. New York: S\&P Global.")
    A(r"\bibitem[中证指数有限公司(2026)]{csindex} 中证指数有限公司. 2026. 《沪深300指数编制方案》及历次样本调整公告. \url{https://www.csindex.com.cn}.")
    A(r"\bibitem[HSI300-Momentum-Strategy(2025)]{hsmo} dongzhaohe321418-lab. 2025. \emph{HSI300-Momentum-Strategy}: 沪深300 Top-20 等权季度动量策略. GitHub. \url{https://github.com/dongzhaohe321418-lab/HSI300-Momentum-Strategy}.")
    A(r"\end{thebibliography}")

    # ------------------------------------------------------------------ appendix
    A(r"\appendix")
    A(r"\section{最新持仓与待实施名单}\label{app:holdings}")
    lh = ctx["latest_holdings"]
    lh_t = pd.DataFrame({"代码": lh["Ticker"], "名称": lh["Name"].map(esc), "Score": lh["Momentum_Score"].map(lambda v: f"{v:.2f}"), "排名": lh["Rank"].astype(int).astype(str),
                         "沪深300权重（代理）": lh["CSI300_Weight"].map(lambda v: pct(v, 2)), "最终权重": lh["Final_Weight"].map(lambda v: pct(v, 2)), "触及上限": lh["Capped"].map({True: "是", False: ""})})
    A(table(lh_t, f"{lh['Effective_Date'].iloc[0]} 实施（参考日 {lh['Reference_Date'].iloc[0]}）的全部 {len(lh)} 只成分股", "tab:holdings", colspec=r"@{}llrrrrl@{}", escape=False, size=r"\scriptsize", longtable=True))
    if ctx["pending_holdings"] is not None:
        ph = ctx["pending_holdings"]
        pt = pd.DataFrame({"代码": ph["Ticker"], "名称": ph["Name"].map(esc), "Score": ph["Momentum_Score"].map(lambda v: f"{v:.2f}"), "排名": ph["Rank"].astype(int).astype(str), "目标权重": ph["Final_Weight"].map(lambda v: pct(v, 2))})
        A(table(pt, f"参考日 {ph['Reference_Date'].iloc[0]} 已计算、将于 {ph['Effective_Date'].iloc[0]} 收盘后实施的 {len(ph)} 只（未进入回测）", "tab:pending", colspec=r"@{}llrrr@{}", escape=False, size=r"\scriptsize", longtable=True))
    A(r"\section{参数与输出文件}")
    par = pd.DataFrame([
        ["Financial Viability", "最近一季 $>0$ 且最近四季合计 $>0$；字段 \\texttt{CONTINUED\\_NETPROFIT} → \\texttt{NETPROFIT}；公告日 $\\leq$ 参考日；不合理公告日 → 法定截止日"],
        ["动量窗口", "参考日前 13 个月 → 前 1 个月；最少 150 个交易日"], ["$Z$ 分数截断", "$\\pm 3$"], ["选样比例", "合格股票的 20\\%（四舍五入）"],
        ["缓冲", "前 80\\% 自动入选；现成分股 120\\% 以内保留"],
        ["权重上限", f"$\\min({cap_abs*100:.0f}\\%,\\,{cap_mult:g}\\,w^P)$；无可行解时倍数逐步加 1（本次运行未触发）"],
        ["调仓", "2 / 8 月最后一个交易日参考；3 / 9 月第三个星期五收盘后实施"],
        ["成本", "佣金、印花税、过户费按历史制度；冲击成本固定 " + pct(cfg['costs'].get('slippage_per_side', 0.0), 2) + " 每边"],
        ["指数基点", f"{cfg['project']['base_level']:.0f}（绩效比较按 \\textyen 100 重设）"],
    ], columns=["参数", "取值"])
    A(table(par, "策略参数（\\texttt{config.yaml}，未做任何收益导向调整）", "tab:params", colspec=r"@{}p{3.2cm}p{11.8cm}@{}", escape=False, header_escape=False))
    files = pd.DataFrame([
        ["output/scores/YYYY-MM-DD.csv", "每个参考日全部沪深300成分股的资格、动量、$Z$ 分数、Momentum Score、排名、权重"],
        ["output/holdings/YYYY-MM-DD.csv", "每次实施的成分股与权重（\\texttt{\\_pending} 为尚未实施）"],
        ["output/performance/index\\_levels.csv", "日频指数点位：沪深300、策略（费前 / 费后）、全收益版本"],
        ["output/performance/strategy\\_ohlc.csv, csi300\\_ohlc.csv", "日 OHLC（策略成交量留空）"],
        ["output/performance/*.csv", "统计、周期、崩塌、调仓摘要与资格审计"],
        ["output/research/*.csv", "本文的归因、HSMO 复现、集中度、漏斗、持续性、行业暴露分析表"],
        ["data/processed/csi300\\_membership\\_intervals.csv", f"点位成分股区间表（{uni['n_codes']} 只证券）"],
    ], columns=["文件", "内容"])
    A(table(files, "主要输出文件", "tab:files", colspec=r"@{}p{7.4cm}p{7.6cm}@{}", escape=False, header_escape=False, size=r"\footnotesize"))
    A(r"\end{document}")
    doc = "\n".join(L).replace("\\subsection{", "\\FloatBarrier\n\\subsection{")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")
    return path


def compile_pdf(tex: Path, engine: str | None = None, extra_args: list[str] | None = None) -> Path | None:
    """Compile with tectonic (preferred) or latexmk -xelatex; returns the PDF path or None."""
    tex = Path(tex)
    if engine is None:
        engine = "tectonic" if shutil.which("tectonic") else "latexmk" if shutil.which("latexmk") else None
    if engine is None:
        log.warning("no LaTeX engine found (install tectonic or TeX Live); .tex written only")
        return None
    if engine == "tectonic":
        cmd = [shutil.which("tectonic"), "-X", "compile", "--keep-logs", *(extra_args or []), tex.name]
    else:
        cmd = [shutil.which("latexmk"), "-xelatex", "-interaction=nonstopmode", "-halt-on-error", *(extra_args or []), tex.name]
    res = subprocess.run(cmd, cwd=str(tex.parent), capture_output=True, text=True)
    pdf = tex.with_suffix(".pdf")
    if res.returncode != 0 or not pdf.exists():
        log.error("LaTeX compilation failed:\n%s", (res.stdout + res.stderr)[-3000:])
        return None
    return pdf
