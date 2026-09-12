"""Presentation-only Q1–Q3 figures/tables from immutable audited outputs.

Run: python -m src.paper_figures.redesign
No solver, training, statistical retesting, or result-file writes.
Style-only adaptation of inspected ModelViz trend templates; project colors and
Chinese typography take precedence. No template smoothing or demo data retained.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/arya-paper-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.plotting import CUMCM_PALETTE as PAL, configure_plots

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "paper/figures/redesign"
GEN = ROOT / "paper/contents/generated"
SOURCES: dict[str, str] = {}
QA: dict[str, object] = {}
NAMES = {"Yesterday": "昨日同刻", "Last Week": "上周同刻",
         "7-day same-slot mean": "近7日同刻均值", "Seasonal/Phase": "相位模型",
         "SSA best config": "SSA", "DLinear": "DLinear",
         "StructuralResidualHybrid": "结构/残差混合",
         "PhaseStructuralFusion": "相位/结构融合"}


def source(path: str) -> Path:
    p = ROOT / path
    SOURCES[path] = hashlib.sha256(p.read_bytes()).hexdigest()
    return p


def csv(path: str) -> pd.DataFrame:
    return pd.read_csv(source(path))


def js(path: str) -> dict:
    return json.loads(source(path).read_text())


def setup() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    GEN.mkdir(parents=True, exist_ok=True)
    configure_plots()
    plt.rcParams.update({"font.sans-serif": ["Songti SC"], "font.size": 9,
                         "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8,
                         "axes.titlesize": 9, "svg.fonttype": "none",
                         "pdf.fonttype": 42, "axes.unicode_minus": False})


def clean(ax, title=None, xlabel=None, ylabel=None):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E2E6EA", lw=0.5)
    ax.set_axisbelow(True)
    if title: ax.set_title(title, loc="left", pad=9)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)


def save(fig, name, contract):
    # AutoLocator may expose labels for ticks outside the displayed limits.
    for ax in fig.axes:
        lo, hi = sorted(ax.get_xlim())
        ax.set_xticks([v for v in ax.get_xticks() if lo <= v <= hi])
        lo, hi = sorted(ax.get_ylim())
        ax.set_yticks([v for v in ax.get_yticks() if lo <= v <= hi])
    fig.canvas.draw()
    # Check text extents against the complete figure, not only axes.
    renderer = fig.canvas.get_renderer()
    outside = []
    for ax in fig.axes:
        for text in [*ax.texts, ax.title, ax.xaxis.label, ax.yaxis.label,
                     *ax.get_xticklabels(), *ax.get_yticklabels()]:
            if not text.get_visible() or not text.get_text(): continue
            b = text.get_window_extent(renderer)
            if b.x0 < -2 or b.y0 < -2 or b.x1 > fig.bbox.width+2 or b.y1 > fig.bbox.height+2:
                outside.append(text.get_text())
    if outside: raise AssertionError(f"Text outside canvas: {name}: {outside}")
    for ext in ("pdf", "svg", "jpg", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=600, facecolor="white")
    svg = FIG / f"{name}.svg"
    svg.write_text("\n".join(x.rstrip() for x in svg.read_text().splitlines())+"\n")
    QA[name] = {"contract": contract, "size_inches": fig.get_size_inches().tolist(),
                "text_outside_canvas": outside, "source_data": list(SOURCES)}
    plt.close(fig)


def table(name, caption, label, headers, rows, spec=None, note=""):
    spec = spec or ("l" + "r"*(len(headers)-1))
    text = ("\\begin{table}[htbp]\n\\centering\\small\n"
            + "\\caption{"+caption+"}\\label{"+label+"}\n"
            + "\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{"+spec+"}\n\\toprule\n"
            + " & ".join(headers)+r" \\"+"\n\\midrule\n")
    text += "\n".join(" & ".join(map(str, row))+r" \\" for row in rows)
    text += "\n\\bottomrule\n\\end{tabular}\n"
    if note: text += "\\par\\vspace{3pt}\\begin{minipage}{0.97\\textwidth}\\footnotesize "+note+"\\end{minipage}\n"
    text += "\\end{table}\n"
    (GEN / f"{name}.tex").write_text(text)


def profiles():
    a = csv("data/processed/C题/actual_10min.csv")
    assert len(a) == 52560 and a.groupby("operating_date").size().eq(144).all()
    date = js("results/tables/data_diagnostics/diagnostics_summary.json")["representative_date"]
    day = a.loc[a.operating_date.eq(date)]
    x = (day.slot.to_numpy()-.5)/6
    f, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), layout="constrained")
    for col, label, color, ls in [("load_kw","负荷",PAL["load"],"-"),
                                 ("pv_actual_kw","光伏",PAL["pv"],"--"),
                                 ("net_load_kw","净负荷",PAL["observed"],":")]:
        axes[0].plot(x, day[col], color=color, ls=ls, lw=1.3, label=label)
    axes[0].axhline(0, color="#777777", lw=.7)
    axes[0].set(xlim=(0,24), ylim=(-4000,10000), xticks=[0,6,12,18,24])
    axes[0].legend(ncol=3, loc="upper center", frameon=False)
    clean(axes[0],f"a  代表日：{date}","时刻（h）","功率（kW）")
    for col,label,color,ls in [("load_kw","负荷",PAL["load"],"-"),("pv_actual_kw","光伏",PAL["pv"],"--")]:
        q = a.groupby("slot")[col].quantile([.25,.5,.75]).unstack()
        axes[1].fill_between(x, q[.25],q[.75],color=color,alpha=.13)
        axes[1].plot(x,q[.5],color=color,ls=ls,label=label,lw=1.4)
    axes[1].set(xlim=(0,24),ylim=(0,10500),xticks=[0,6,12,18,24])
    axes[1].legend(loc="upper left",frameon=False)
    clean(axes[1],"b  全年同刻中位数与四分位带","时刻（h）","功率（kW）")
    save(f,"fig_data_profiles","Representative date fixed by saved audit; all 365 days; median and IQR across days, not CI.")
    f,axes=plt.subplots(1,2,figsize=(7.1,2.6),layout="constrained")
    months=pd.to_datetime(a.operating_date).dt.month
    for m,label,color,ls,marker in [([3,4,5],"春",PAL["pv"],"-","o"),([6,7,8],"夏",PAL["price"],"--","s"),([9,10,11],"秋",PAL["soc"],"-.","^"),([12,1,2],"冬",PAL["load"],":","D")]:
        v=a.loc[months.isin(m)].groupby("slot").pv_actual_kw.mean()
        axes[0].plot(x,v,label=label,color=color,ls=ls,marker=marker,markevery=18,ms=3,lw=1.2)
    axes[0].legend(ncol=4,loc="upper center",frameon=False,handlelength=1.7,columnspacing=.8)
    axes[0].set(xlim=(0,24),ylim=(0,10200),xticks=[0,6,12,18,24])
    clean(axes[0],"a  各季节平均光伏日轮廓","时刻（h）","光伏功率（kW）")
    # Same pairwise Pearson definition as the existing diagnostic script.
    pv=a.pv_actual_kw.to_numpy()
    for daylight,label,color,ls in [(False,"全序列",PAL["load"],"-"),(True,"两端均有光伏出力",PAL["pv"],"--")]:
        values=[]
        for lag in range(1009):
            left,right=(pv,pv) if lag==0 else (pv[lag:],pv[:-lag])
            mask=(left>0)&(right>0) if daylight else np.ones(len(left),dtype=bool)
            values.append(np.corrcoef(left[mask],right[mask])[0,1] if mask.sum()>2 else np.nan)
        axes[1].plot(np.arange(1009)/144,values,color=color,ls=ls,lw=1,label=label)
    axes[1].set(xlim=(0,7),ylim=(-1.06,1.23),xticks=range(8),yticks=[-1,-.5,0,.5,1])
    axes[1].axhline(0,color="#777777",lw=.7)
    axes[1].legend(loc="upper center",ncol=2,frameon=False,fontsize=7)
    clean(axes[1],"b  完整日、周时滞相关","滞后天数","Pearson相关系数")
    save(f,"fig_periodicity","Full correlation range retained; gaps only where paired daylight samples do not exist; no diagnostic model rerun.")


def q1():
    a=csv("results/problem1/tables/table_p1_dispatch.csv")
    s=js("results/problem1/tables/table_p1_summary.json")
    assert len(a)==144 and abs(s['optimal_cost_yuan']-35126.948591)<1e-5
    f,ax=plt.subplots(4,1,figsize=(7.1,5.8),sharex=True,layout="constrained",gridspec_kw={"height_ratios":[1.5,1,1.3,.7]})
    x=np.arange(145)/6
    for col,label,color,ls in [("load_kw","负荷",PAL['load'],'-'),("pv_forecast_kw","给定光伏",PAL['pv'],'--'),("grid_purchase_kw","计划购电",PAL['neutral'],':')]:
        ax[0].stairs(a[col],x,label=label,color=color,lw=1.2,linestyle=ls)
    ax[0].set_ylim(0,11200); ax[0].legend(ncol=3,loc="upper center",frameon=False)
    clean(ax[0],"a  供需功率",ylabel="功率（kW）")
    ax[1].stairs(a.charge_kw,x,color=PAL['charge'],label="充电（正）",lw=1.1)
    ax[1].stairs(-a.discharge_kw,x,color=PAL['discharge'],ls="--",label="放电（负向展示）",lw=1.1)
    ax[1].axhline(0,color="#555555",lw=.6); ax[1].set_ylim(-6500,7200)
    ax[1].legend(ncol=2,loc="upper center",frameon=False)
    clean(ax[1],"b  储能充放电",ylabel="功率（kW）")
    e=np.r_[a.storage_start_kwh.iloc[0],a.storage_end_kwh]
    ax[2].plot(x,e,color=PAL['soc'],lw=1.4)
    for bound in [1200,10800]: ax[2].axhline(bound,color="#999999",ls="--",lw=.8)
    ax[2].set_ylim(0,13500)
    clean(ax[2],"c  储能电量及安全边界",ylabel="电量（kWh）")
    inset=ax[2].inset_axes([.60,.55,.30,.40])
    mask=x>=21
    inset.plot(x[mask],e[mask],color=PAL['soc'],lw=1.1,marker="o",ms=2,markevery=6)
    inset.set(xlim=(21,24),ylim=(0,6500),xticks=[21,24],yticks=[1200,6000])
    inset.tick_params(labelsize=6); inset.set_title("日末21-24 h细节",fontsize=7,pad=2)
    ax[3].stairs(a.price_yuan_per_kwh,x,color=PAL['price'],lw=1.2)
    ax[3].set(ylim=(0,1.7),xlim=(0,24),xticks=[0,4,8,12,16,20,24])
    clean(ax[3],"d  题设分时电价","时刻（h）","电价（元/kWh）")
    save(f,"fig_q1_dispatch","144 immutable intervals plus 145 energy boundaries; no uncertainty; 21–24 h inset from identical state trajectory.")
    table("q1_summary","问题一经济收益与物理核验","tab:p1-summary",
          ["指标","无储能对照","正式MILP","改善或核验"],[
              ["购电费用/元",f"{s['baseline_cost_yuan']:.3f}",f"{s['optimal_cost_yuan']:.6f}",f"节省{s['cost_saving_yuan']:.3f}"],
              ["节费率", "--", "--",f"{s['cost_saving_percent']:.3f}\\%"],
              ["外网购电量/kWh",f"{s['baseline_grid_energy_kwh']:.3f}",f"{s['total_grid_energy_kwh']:.3f}",f"减少{s['baseline_grid_energy_kwh']-s['total_grid_energy_kwh']:.3f}"],
              ["弃光量/kWh",f"{s['baseline_spill_energy_kwh']:.3f}","0","全部消纳"],
              ["储能电量/kWh","不适用","1200--10800","首末均为6000"],
              ["最大功率平衡残差/kW","--",r"$4.0\times10^{-5}$","通过"],
              ["最大电量递推残差/kWh","--",r"$4.7\times10^{-4}$","通过"]],spec="lrrl")
    lp=csv("results/problem1/tables/q1_lp_relaxation_comparison.csv")
    table("q1_lp","附件1单日算例的MILP与LP数值对照","tab:p1-lp-relaxation",
          ["模型","目标值/元","最大同时充放电/kW","单次用时/s"],
          [[r.formulation,f"{r.objective_cost_yuan:.6f}",f"{r.maximum_simultaneous_charge_discharge_kw:.0f}",f"{r.solver_runtime_seconds:.3f}"] for r in lp.itertuples()],
          note="两次求解使用相同输入，间隙为0元；用时为已保存的单次测量，不表示重复实验平均值。")
    f,axes=plt.subplots(1,3,figsize=(7.1,2.4),sharey=True,layout="constrained")
    for ax,name,title,color,base in zip(axes,["efficiency","capacity","power"],["a  单程效率","b  容量倍数","c  功率上限倍数"],[PAL['load'],PAL['pv'],PAL['price']],[.9,1,1]):
        data=csv(f"results/problem1/tables/q1_{name}_sensitivity.csv")
        assert data.solver_status.eq('Optimal').all() and data.physical_invariants_pass.all()
        ax.plot(data.parameter_value,data.optimal_cost_yuan/1e4,color=color,marker='o',ms=4,lw=1.2)
        orig=data.loc[np.isclose(data.parameter_value,base)].iloc[0]
        ax.plot(base,orig.optimal_cost_yuan/1e4,marker='s',mfc='white',mec='black',ms=6)
        for row in data.itertuples():
            ax.annotate(f"{row.optimal_cost_yuan/1e4:.3f}",(row.parameter_value,row.optimal_cost_yuan/1e4),xytext=(0,8),textcoords='offset points',ha='center',fontsize=7)
        ax.set(ylim=(3.25,4.25),xticks=data.parameter_value)
        ax.margins(x=.15); clean(ax,title,xlabel=title[3:])
    axes[0].set_ylabel("购电费用（万元）")
    save(f,"fig_q1_sensitivity","Four deterministic cases per panel; explicit local cost scale; no error bars; hollow square baseline.")


def q2():
    a=csv("results/tables/forecasting/forecast_model_final_comparison.csv")
    a=a.loc[a.model.isin(NAMES)].copy()
    assert len(a)==8 and a.loc[a.selected_by_january,'model'].tolist()==['7-day same-slot mean']
    table("q2_forecasts","光伏预测的因果比较：一月选型，2--12月检验","tab:q2-forecast-comparison",
          ["模型","验证MAE","验证RMSE","正式期MAE","正式期RMSE"],
          [[NAMES[r.model],*[f"{getattr(r,c):.3f}" for c in ['validation_mae_kw','validation_rmse_kw','oos_mae_kw','oos_rmse_kw']]] for r in a.itertuples()],
          note="误差单位均为kW。验证期为1月18--31日；模型仅按验证RMSE最小、MAE并列判据选择。")
    daily=csv("results/problem2/tables/q2_forecast_daily_mae.csv").pivot(index='date',columns='model',values='daily_mae')
    comps=['Yesterday','Last Week','SSA best config','DLinear','PhaseStructuralFusion']
    f,ax=plt.subplots(figsize=(7.1,2.4),layout='constrained')
    for i,m in enumerate(comps):
        diff=daily['7-day same-slot mean']-daily[m]
        assert diff.notna().all() and len(diff)==334
        q=diff.quantile([.25,.5,.75]).to_numpy()
        ax.errorbar(q[1],i,xerr=[[q[1]-q[0]],[q[2]-q[1]]],fmt=['o','s','^','D','v'][i],color=PAL['load'],ecolor=PAL['primary_light'],capsize=4,ms=5,lw=2)
    ax.axvline(0,color='#555555',ls='--',lw=.8)
    ax.set(yticks=range(5),yticklabels=[NAMES[x] for x in comps],ylim=(4.6,-.6),xlim=(-450,60))
    clean(ax,"334个运营日的配对误差差值","日MAE差：近7日均值 - 对照（kW）")
    save(f,"fig_q2_daily_difference","Point=median daily paired MAE difference; bars=25th–75th percentiles of 334 days, NOT CI/SE. Full comparisons in table.")
    s=js("results/problem2/tables/table_p2_summary.json")
    table("q2_results","问题二期望成本主方案的费用与能量核验","tab:q2-results",
          ["指标","模型内期望","实际结算"],[
              ["计划购电费/元",f"{s['planned_purchase_cost_yuan']:.6f}",f"{s['planned_purchase_cost_yuan']:.6f}"],
              ["紧急购电费/元",f"{s['expected_emergency_cost_yuan']:.6f}",f"{s['realized_emergency_cost_yuan']:.6f}"],
              ["总购电费/元",f"{s['expected_total_cost_yuan']:.6f}",f"{s['realized_total_cost_yuan']:.6f}"],
              ["紧急购电量/kWh",f"{s['expected_emergency_energy_kwh']:.6f}",f"{s['realized_emergency_energy_kwh']:.6f}"],
              ["未使用计划电量/kWh",f"{s['expected_unused_planned_grid_energy_kwh']:.3f}",f"{s['realized_unused_planned_grid_energy_kwh']:.3f}"],
              ["实际储能电量/kWh","--","1200--10800"],
              ["未吸收放电/kWh","--","0"]],
          note="334/334日求解最优；风险权重为0。期望与实际为两种评价口径，并非互为基准的两种策略。延续价值不计入结算。")
    r=csv("results/problem2/q2_risk_sweep.csv").sort_values('lambda')
    f,ax=plt.subplots(figsize=(7.1,2.55),layout='constrained')
    ax.plot(r.cvar_cost_yuan/1e4,r.expected_operating_cost_yuan/1e4,color=PAL['load'],lw=1.3,marker='o',ms=5)
    for i,row in enumerate(r.itertuples(index=False)):
        lam=float(r.iloc[i]['lambda']); x=float(r.iloc[i].cvar_cost_yuan)/1e4; y=float(r.iloc[i].expected_operating_cost_yuan)/1e4
        ax.annotate(rf"$\lambda_{{\mathrm{{r}}}}={lam:g}$",(x,y),xytext=(8,8),textcoords='offset points',fontsize=8)
    ax.plot(r.cvar_cost_yuan.iloc[1]/1e4,r.expected_operating_cost_yuan.iloc[1]/1e4,'s',mfc='white',mec=PAL['price'],ms=8)
    ax.set(xlim=(70,610),ylim=(1420,1535))
    clean(ax,"预设风险偏好的成本—风险权衡","日紧急购电费用CVaR之和（万元）","期望运行费用（万元）")
    save(f,"fig_q2_risk","Five fixed lambda preferences; daily CVaR summed across 334 days, not CVaR of annual total; no uncertainty bars.")


def q3():
    bins=['1-3h','4-6h','7-12h','13-18h','19-24h']
    a=csv('results/problem3/tables/q3_official_forecast_by_lead.csv')
    a=a.loc[a.subset.eq('全部时段')].set_index('lead_bin').loc[bins]
    c=csv('results/problem3/tables/q3_forecast_method_comparison.csv')
    f,axes=plt.subplots(1,2,figsize=(7.1,2.65),layout='constrained')
    for col,label,color,ls,m in [('mae_kw','MAE',PAL['pv'],'-','o'),('rmse_kw','RMSE',PAL['load'],'--','s')]:
        axes[0].plot(range(5),a[col],color=color,ls=ls,marker=m,label=label,ms=4)
    for method,color,ls,m in [('官方预测',PAL['neutral'],'--','s'),('提前期分箱融合',PAL['load'],'-','o')]:
        rows=c.loc[c.period.eq('evaluation')&c.scope_type.eq('lead_bin')&c.method.eq(method)].set_index('scope_value').loc[bins]
        axes[1].plot(range(5),rows.rmse_kw,color=color,ls=ls,marker=m,label=method,ms=4)
    for ax in axes:
        ax.set(xticks=range(5),xticklabels=['1-3','4-6','7-12','13-18','19-24'],ylim=(0,850))
        ax.legend(loc='upper left',frameon=False)
    clean(axes[0],"a  全年官方预测", "提前期（h）","误差（kW）")
    clean(axes[1],"b  同口径2至12月样本外比较","提前期（h）","RMSE（kW）")
    save(f,"fig_q3_lead","Left all 35004 matched hourly targets; right same saved OOS samples for both methods; aggregated errors, not replicate intervals.")
    s=csv('results/problem3/tables/q3_schedule_comparison.csv').set_index('schedule').loc[['S0','S1','S2','S3']]
    v=csv('results/problem3/tables/q3_voi.csv').set_index('schedule').loc[s.index]
    sens=csv('results/problem3/tables/q3_settlement_mode_sensitivity.csv')
    js('results/problem3/tables/q3_forecast_selection.json')
    source('results/problem3/tables/q3_daily_economic_results.csv')
    assert s.realized_total_cost_yuan.idxmin()=='S2' and v.loc['S3','incremental_voi_yuan']<0
    f,axes=plt.subplots(1,3,figsize=(7.1,2.75),layout='constrained',gridspec_kw={'width_ratios':[.85,1,1.25]})
    costs=s.realized_total_cost_yuan.to_numpy()/1e4
    colors=[PAL['neutral'],PAL['primary_light'],PAL['pv'],PAL['load']]
    for i,(val,color) in enumerate(zip(costs,colors)):
        axes[0].plot(i,val,marker=['o','s','D','^'][i],color=color,ms=6)
        axes[1].plot(i,val,marker=['o','s','D','^'][i],color=color,ms=6)
        axes[1].annotate(f"{val:.2f}",(i,val),xytext=(0,7),textcoords='offset points',ha='center',fontsize=7)
    axes[0].set(ylim=(0,1700),xlim=(-.5,3.5),xticks=range(4),xticklabels=s.index)
    axes[1].set(ylim=(1446,1482),xlim=(-.5,3.5),xticks=range(4),xticklabels=s.index)
    axes[1].text(.05,.95,'S2最低',transform=axes[1].transAxes,va='top',fontsize=8,color=PAL['pv'])
    clean(axes[0],"a  费用全尺度",ylabel="实际费用（万元）")
    clean(axes[1],"b  明示局部放大",ylabel="实际费用（万元）")
    vals=v.incremental_voi_yuan.to_numpy()[1:]/1e4
    axes[2].axhline(0,color='#555555',lw=.8)
    for i,(val,color) in enumerate(zip(vals,[PAL['load'],PAL['pv'],PAL['bad']])):
        axes[2].vlines(i,0,val,color=color,lw=1.8)
        axes[2].plot(i,val,marker=['o','D','v'][i],color=color,ms=6)
        axes[2].annotate(f"{val:+.4f}",(i,val),xytext=(0,8 if val>=0 else -17),textcoords='offset points',ha='center',fontsize=7)
    axes[2].set(xlim=(-.55,2.55),ylim=(-3,16),xticks=range(3),xticklabels=['06:00','12:00','18:00'])
    clean(axes[2],"c  新增预报的边际价值",ylabel="增量VOI（万元）")
    save(f,"fig_q3_economics","Identical four audited schedules; zero-origin overview plus explicit detail; negative VOI not clipped; no errorbars.")
    table('q3_results','问题三四种更新时间表的实际结算与信息价值','tab:q3-schedule-comparison',
          ['安排','实际总费用/元','紧急电量/kWh','相对S0节费/元'],
          [[i,f"{s.loc[i,'realized_total_cost_yuan']:.6f}",f"{s.loc[i,'realized_emergency_energy_kwh']:.3f}",f"{v.loc[i,'voi_vs_s0_yuan']:.3f}"] for i in s.index],
          note="正式期为2月1日至12月31日。S0使用问题三的官方预报融合信息，不是问题二主方案的复现。")


def main():
    setup(); profiles(); q1(); q2(); q3()
    for p,digest in SOURCES.items():
        assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest
    (GEN/'redesign_sources.json').write_text(json.dumps({'source_hashes':SOURCES,'figures':QA},ensure_ascii=False,indent=2)+'\n')
    print(f"Generated {len(QA)} figures; {len(SOURCES)} sources unchanged.")


if __name__=='__main__': main()
