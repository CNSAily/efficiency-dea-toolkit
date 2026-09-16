# -*- coding: utf-8 -*-
"""visualize.py —— 效率评估结果可视化模块。

提供效率研究中高频出现的四类图：
    1. 时序趋势图    —— 效率随年份的均值演变（"整体跃升"）
    2. 核密度曲线    —— 效率分布形态的时空演化（"波峰右移"）
    3. 区域对比图    —— 分区域效率对比（"区域非对称赶超"）
    4. 马尔科夫转移图 —— 效率状态转移概率（"路径依赖 / 锁定效应"）

统一约定：效率数据以长表 DataFrame 传入，列含 entity、year、效率值列。
所有函数返回 (fig, ax)，便于在 Notebook 中继续调整或直接保存。
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def setup_chinese_font():
    """设置 matplotlib 中文字体，避免图表中文显示为方框。"""
    for name in ("Microsoft YaHei", "SimHei", "PingFang SC"):
        try:
            matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return name
        except Exception:
            continue
    return None


def plot_trend(eff_df: pd.DataFrame, value_col: str,
               year_col: str = "year", title: str = "效率年度趋势",
               ylabel: str = "平均效率") -> tuple:
    """时序折线：效率均值随年份变化。"""
    yearly = eff_df.groupby(year_col)[value_col].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(yearly[year_col], yearly[value_col], marker="o", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("年份")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    return fig, ax


def plot_kernel(eff_df: pd.DataFrame, value_col: str, years: list[int],
                year_col: str = "year", title: str = "效率核密度分布演化") -> tuple:
    """核密度曲线：多个年份的效率分布形态叠加，观察波峰移动。"""
    from scipy.stats import gaussian_kde

    fig, ax = plt.subplots(figsize=(7, 4))
    x_grid = np.linspace(0, 1, 200)
    cmap = plt.cm.viridis(np.linspace(0.2, 0.9, len(years)))
    for y, c in zip(years, cmap):
        vals = eff_df.loc[eff_df[year_col] == y, value_col].dropna().to_numpy()
        if vals.size < 3:
            continue
        kde = gaussian_kde(vals, bw_method=0.3)
        ax.plot(x_grid, kde(x_grid), label=str(y), color=c, linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("效率值")
    ax.set_ylabel("密度")
    ax.legend(title="年份")
    return fig, ax


def plot_region_bar(eff_df: pd.DataFrame, value_col: str, region_col: str,
                    title: str = "区域效率对比", ylabel: str = "平均效率") -> tuple:
    """分区域效率条形图。"""
    region = eff_df.groupby(region_col)[value_col].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(region.index.astype(str), region.values)
    ax.set_title(title)
    ax.set_xlabel(ylabel)
    return fig, ax


def markov_transition(eff_df: pd.DataFrame, value_col: str,
                      entity_col: str = "entity", year_col: str = "year",
                      n_state: int = 4) -> pd.DataFrame:
    """马尔科夫转移矩阵：效率离散为 n_state 个状态后的转移概率。

    状态按各年合并样本的分位数划分（低→高）。矩阵元素 P[i,j] =
    从状态 i 转移到状态 j 的比例（按"同一实体相邻年份"统计）。
    """
    df = eff_df.sort_values([entity_col, year_col]).copy()
    all_vals = df[value_col].to_numpy()
    bins = np.percentile(all_vals, np.linspace(0, 100, n_state + 1))
    bins[0], bins[-1] = -np.inf, np.inf
    df["state"] = np.digitize(df[value_col].to_numpy(), bins) - 1

    trans = np.zeros((n_state, n_state))
    for _, g in df.groupby(entity_col):
        g = g.sort_values(year_col)
        states = g["state"].to_numpy()
        for i in range(len(states) - 1):
            trans[states[i], states[i + 1]] += 1

    row_sum = trans.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    P = trans / row_sum
    return pd.DataFrame(P, index=[f"S{i+1}" for i in range(n_state)],
                        columns=[f"S{i+1}" for i in range(n_state)]).round(3)


def plot_markov(eff_df: pd.DataFrame, value_col: str,
                entity_col: str = "entity", year_col: str = "year",
                n_state: int = 4, title: str = "效率状态转移概率") -> tuple:
    """马尔科夫转移矩阵热力图。"""
    P = markov_transition(eff_df, value_col, entity_col, year_col, n_state)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(P.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n_state), P.columns)
    ax.set_yticks(range(n_state), P.index)
    ax.set_title(title)
    ax.set_xlabel("当期状态")
    ax.set_ylabel("上一期状态")
    fig.colorbar(im, ax=ax, label="概率")
    for i in range(n_state):
        for j in range(n_state):
            ax.text(j, i, f"{P.iloc[i, j]:.2f}", ha="center", va="center",
                    color="white" if P.iloc[i, j] > 0.5 else "black", fontsize=9)
    return fig, ax
