# -*- coding: utf-8 -*-
"""data.py —— 面板数据预处理模块。

面向经管效率评估（DEA）的通用数据准备：
    1. 读取长表格式面板数据（实体 × 年份 × 指标）
    2. 校验投入/产出数据的非负性与维度
    3. 处理缺失值（可选：均值插补 / 删除）
    4. 无量纲化 / 归一化（DEA 效率对线性尺度不敏感，但两阶段
       结构下为保持数值稳定常做 0-1 归一化）

设计原则：每个函数职责单一、可独立复用，方便面试逐层讲解。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. 数据读取与校验
# ---------------------------------------------------------------------------
def load_panel(path: str) -> pd.DataFrame:
    """读取长表格式面板数据（一行 = 一个实体在某一年的一条记录）。

    约定列名（见 examples/demo_data.csv）：
        entity   : 实体标识（如省份）
        year     : 年份
        其余列    : 各投入/产出/机制变量指标
    """
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def validate_positive(df: pd.DataFrame, cols: Sequence[str]) -> None:
    """校验指定列为严格正值（DEA 投入/产出必须 > 0）。

    违规时抛出 ValueError，附带具体列与最小值的诊断信息，
    避免"数值异常导致效率算出 inf/NaN"这类难以排查的问题。
    """
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce")
        if v.isna().any():
            raise ValueError(f"列 {c!r} 存在无法转换为数值的缺失/异常值")
        if (v <= 0).any():
            raise ValueError(f"列 {c!r} 存在非正值（DEA 要求 > 0），最小值为 {v.min()}")


def impute_mean(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """按实体分组、按列做均值插补（针对少量缺失的面板数据）。

    返回副本，不修改原 DataFrame。
    """
    out = df.copy()
    for c in cols:
        out[c] = out[c].fillna(out.groupby("entity")[c].transform("mean"))
    return out


# ---------------------------------------------------------------------------
# 2. 无量纲化 / 归一化
# ---------------------------------------------------------------------------
def minmax_scale(series: pd.Series, lo: float = 1e-3, hi: float = 1.0) -> pd.Series:
    """把一列数值线性缩放到 [lo, hi]，保留相对大小与分布形态。

    DEA 效率是投入产出"比例"的度量，对正线性变换不变；这里归一化
    主要用于：① 统一不同量纲指标；② 提升两阶段 DEA 线性规划的数值稳定性。
    """
    s = pd.to_numeric(series, errors="coerce")
    smin, smax = s.min(), s.max()
    if smax == smin:
        # 常量列：全部落到区间中点，避免除零
        return pd.Series(lo + (hi - lo) / 2, index=series.index)
    return (s - smin) / (smax - smin) * (hi - lo) + lo


def normalize_panel(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """对指定指标列做面板级归一化（跨全部实体/年份一起缩放）。

    返回副本。
    """
    out = df.copy()
    for c in cols:
        out[c] = minmax_scale(out[c])
    return out


# ---------------------------------------------------------------------------
# 3. 两阶段 DEA 数据装配
# ---------------------------------------------------------------------------
def assemble_dea_inputs(df: pd.DataFrame,
                        stage1_inputs: Sequence[str],
                        stage1_outputs: Sequence[str],   # 第一阶段的产出 = 第二阶段的投入
                        stage2_outputs: Sequence[str],
                        entity_col: str = "entity",
                        year_col: str = "year") -> tuple[list[np.ndarray], list]:
    """把长表数据装配成两阶段 DEA 所需的矩阵与元信息。

    两阶段网络结构（参考 Lafuente et al. 2024 的分层思想）：
        第一阶段：stage1_inputs -> stage1_outputs（中间产出）
        第二阶段：stage1_outputs（作为投入）-> stage2_outputs

    返回 (matrices, index)：
        matrices = [X1, Z, Y2]
            X1 : 第一阶段投入矩阵  (N, n_x1)
            Z  : 中间产出矩阵      (N, n_z)   —— 既是阶段一产出，也是阶段二投入
            Y2 : 第二阶段产出矩阵  (N, n_y2)
        index    : 每个决策单元（DMU）的标识（如 "北京_2012"），便于回溯
    """
    require_cols = list(stage1_inputs) + list(stage1_outputs) + list(stage2_outputs)
    missing = [c for c in require_cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少列：{missing}")

    index = (df[entity_col].astype(str) + "_" + df[year_col].astype(str)).tolist()

    X1 = df[list(stage1_inputs)].to_numpy(dtype=float)
    Z = df[list(stage1_outputs)].to_numpy(dtype=float)
    Y2 = df[list(stage2_outputs)].to_numpy(dtype=float)

    # 数值合法性兜底
    for name, mat in (("stage1_inputs", X1), ("stage1_outputs", Z), ("stage2_outputs", Y2)):
        if np.any(~np.isfinite(mat)):
            raise ValueError(f"{name} 存在 inf/NaN，请先清洗")
        if np.any(mat <= 0):
            raise ValueError(f"{name} 存在非正值，DEA 要求投入产出 > 0")

    return [X1, Z, Y2], index


# ---------------------------------------------------------------------------
# 4. 演示数据生成（不依赖真实论文数据，可复现）
# ---------------------------------------------------------------------------
def make_demo_data(n_entity: int = 31,
                   n_year: int = 11,
                   seed: int = 42) -> pd.DataFrame:
    """生成一份模拟的"农业数字创业生态系统效率"面板数据（长表）。

    结构对位真实研究口径，但数值为随机模拟、无真实含义，仅作演示：
        entity   : 省份（编号省1..省N，避免使用真实行政区划名）
        year     : 年份（2012 起）
        stage1_inputs   : 第一阶段投入（资源禀赋类，如"人力资本""财政支农"）
        stage1_outputs  : 中间产出（数字基础设施类，作为第二阶段投入）
        stage2_outputs  : 最终产出（数字创业活动类）
        income   : 门槛变量（农村居民人均可支配收入，用于机制分析）
    """
    rng = np.random.default_rng(seed)
    entities = [f"省{i:02d}" for i in range(1, n_entity + 1)]
    years = list(range(2012, 2012 + n_year))

    rows = []
    # 每个实体一个固定"发展水平"，模拟省份间差异
    base = rng.uniform(0.5, 1.5, size=n_entity)

    for e, b in zip(entities, base):
        for t, y in enumerate(years):
            # 年份趋势：整体逐年上升，叠加个体扰动
            trend = 1 + 0.06 * t
            noise = rng.normal(0, 0.08)
            # 门槛变量（农村居民人均可支配收入）
            income = max(0.2, b * (1 + 0.05 * t) + noise)
            # 收入对最终产出（创业活跃度）存在"门槛非线性"驱动：
            # 收入低于门槛 gamma 时斜率平缓，跨过后斜率变陡（模拟真实门槛机制）
            gamma = 1.3
            if income <= gamma:
                drive = 0.4 * income
            else:
                drive = 0.4 * gamma + 1.0 * (income - gamma)
            output = b * trend * (1.0 + drive) * rng.uniform(0.9, 1.1)
            rows.append({
                "entity": e,
                "year": y,
                # 第一阶段投入（资源禀赋）
                "人力资本": max(0.1, b * trend * rng.uniform(0.8, 1.2)),
                "财政支农": max(0.1, b * trend * rng.uniform(0.8, 1.2)),
                # 中间产出（数字基础设施，兼作第二阶段投入）
                "数字基建": max(0.1, b * trend * rng.uniform(0.8, 1.2)),
                # 第二阶段产出（数字创业活动，受收入门槛驱动）
                "创业活跃度": max(0.1, output),
                # 门槛变量
                "收入": income,
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["entity", "year"]).reset_index(drop=True)
    return df
