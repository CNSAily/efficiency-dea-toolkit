# -*- coding: utf-8 -*-
"""pipeline.py —— 端到端效率评估流程。

把四个模块串成一条命令：
    数据预处理 -> 两阶段 DEA 效率 -> 机制分析（门槛/分位数）-> 可视化

典型用法：
    from dea_toolkit import pipeline
    res = pipeline.run_end_to_end(
        df,
        stage1_inputs=["人力资本", "财政支农"],
        stage1_outputs=["数字基建"],
        stage2_outputs=["创业活跃度"],
        x_var="收入", q_var="收入",
        output_dir="output",
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from . import data as dtk_data
from . import dea as dtk_dea
from . import mechanism as dtk_mech
from . import visualize as dtk_viz


def run_end_to_end(df: pd.DataFrame,
                   stage1_inputs: Sequence[str],
                   stage1_outputs: Sequence[str],
                   stage2_outputs: Sequence[str],
                   x_var: str,
                   q_var: str | None = None,
                   entity_col: str = "entity",
                   year_col: str = "year",
                   region_col: str | None = None,
                   output_dir: str = "output",
                   orientation: str = "output",
                   returns: str = "crs",
                   frontier: str = "global",
                   ) -> dict:
    """执行完整效率评估流程，返回结构化结果字典。

    参数：
        df               : 长表面板数据
        stage1_inputs    : 第一阶段投入列
        stage1_outputs   : 中间产出列（兼作第二阶段投入）
        stage2_outputs   : 第二阶段产出列
        x_var            : 核心解释变量（机制分析用）
        q_var            : 门槛变量（默认与 x_var 相同）
        region_col       : 可选，区域分组列（区域对比图用）
        output_dir       : 结果与图表输出目录

    返回字典：eff_df（含效率值的面板）、dea（各阶段效率）、
              threshold（门槛结果）、quantile（分位数结果）、
              markov（转移矩阵）、figs（图表路径列表）。
    """
    q_var = q_var or x_var
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtk_viz.setup_chinese_font()

    # ---- 1. 归一化（全局统一，保证逐年口径一致） ----
    all_metrics = list(stage1_inputs) + list(stage1_outputs) + list(stage2_outputs)
    df_norm = dtk_data.normalize_panel(df, all_metrics)

    # ---- 2. 两阶段 DEA ----
    eff_df = df.copy()
    eff_df["stage1_eff"] = np.nan
    eff_df["stage2_eff"] = np.nan
    eff_df["system_eff"] = np.nan

    def _run_dea(sub):
        (X1, Z, Y2), _ = dtk_data.assemble_dea_inputs(
            sub, stage1_inputs, stage1_outputs, stage2_outputs,
            entity_col=entity_col, year_col=year_col)
        return dtk_dea.two_stage_eff(X1, Z, Y2,
                                     orientation=orientation, returns=returns)

    if frontier == "yearly":
        # 逐年前沿：每年单独构造前沿，效率值落在更合理的区间，适合看趋势
        for yr, sub in df_norm.groupby(year_col):
            e = _run_dea(sub)
            eff_df.loc[sub.index, "stage1_eff"] = e["stage1"]
            eff_df.loc[sub.index, "stage2_eff"] = e["stage2"]
            eff_df.loc[sub.index, "system_eff"] = e["system"]
    else:
        # 全局前沿：所有年份混合构造前沿，可跨期直接比较（默认）
        e = _run_dea(df_norm)
        eff_df["stage1_eff"] = e["stage1"]
        eff_df["stage2_eff"] = e["stage2"]
        eff_df["system_eff"] = e["system"]

    # ---- 3. 机制分析 ----
    threshold = dtk_mech.panel_threshold(
        y=eff_df["system_eff"].to_numpy(),
        x=eff_df[x_var].to_numpy(),
        q=eff_df[q_var].to_numpy(),
        entity=eff_df[entity_col].to_numpy(),
    )
    Xq = eff_df[[x_var]].to_numpy()
    quantile = dtk_mech.quantile_regression(eff_df["system_eff"].to_numpy(), Xq)

    # ---- 4. 可视化 ----
    figs = []
    fig, _ = dtk_viz.plot_trend(eff_df, "system_eff", year_col=year_col)
    p = out_dir / "trend.png"; fig.savefig(p, dpi=150, bbox_inches="tight"); plt_close(fig); figs.append(str(p))

    fig, _ = dtk_viz.plot_kernel(eff_df, "system_eff",
                                 years=sorted(eff_df[year_col].unique())[::2],
                                 year_col=year_col)
    p = out_dir / "kernel.png"; fig.savefig(p, dpi=150, bbox_inches="tight"); plt_close(fig); figs.append(str(p))

    if region_col:
        fig, _ = dtk_viz.plot_region_bar(eff_df, "system_eff", region_col)
        p = out_dir / "region.png"; fig.savefig(p, dpi=150, bbox_inches="tight"); plt_close(fig); figs.append(str(p))

    markov = dtk_viz.markov_transition(eff_df, "system_eff", entity_col, year_col)
    fig, _ = dtk_viz.plot_markov(eff_df, "system_eff", entity_col, year_col)
    p = out_dir / "markov.png"; fig.savefig(p, dpi=150, bbox_inches="tight"); plt_close(fig); figs.append(str(p))

    return {
        "eff_df": eff_df,
        "dea": {
            "stage1": eff_df["stage1_eff"].tolist(),
            "stage2": eff_df["stage2_eff"].tolist(),
            "system": eff_df["system_eff"].tolist(),
        },
        "threshold": threshold,
        "quantile": quantile,
        "markov": markov,
        "figs": figs,
    }


def plt_close(fig):
    """关闭 matplotlib 图，释放内存（批量出图时避免 warning）。"""
    import matplotlib.pyplot as plt
    plt.close(fig)
