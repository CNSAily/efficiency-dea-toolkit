# -*- coding: utf-8 -*-
"""mechanism.py —— 效率影响机制分析模块。

在算出效率之后，研究者通常关心"什么因素驱动了效率提升、驱动作用
是否非线性"。本模块实现两个通用方法：

    1. 面板门槛回归（Hansen, 1999）：检验解释变量对效率的驱动是否
       存在"门槛非线性"——即存在一个临界值 γ，跨过它前后斜率显著不同。
    2. 分位数回归（Koenker & Bassett, 1978）：刻画解释变量在效率分布
       的不同分位点上的异质效应（低效率组 vs 高效率组，谁的边际更强）。

设计原则：面板门槛的核心算法（within 变换 + 网格搜索 + 最小 SSE）手写，
不依赖黑盒包，面试可逐步讲解。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. 面板固定效应门槛回归（Hansen 1999）
# ---------------------------------------------------------------------------
def _within_transform(v: np.ndarray, group: np.ndarray) -> np.ndarray:
    """组内去均值（within transformation），用于消除个体固定效应。

    对每个实体，用其观测值减去该实体的组内均值。
    这是面板固定效应模型"去均值后 OLS"的标准第一步。
    """
    df = pd.DataFrame({"v": v, "g": group})
    mean = df.groupby("g")["v"].transform("mean")
    return (df["v"] - mean).to_numpy()


def _ols_ssr(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float]:
    """普通最小二乘，返回 (系数, 残差平方和 SSE)。

    X 不含截距列（调用方按需拼入）。用 lstsq 求解，稳定且无额外依赖。
    """
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    return coef, float(resid @ resid)


def panel_threshold(y: np.ndarray,
                    x: np.ndarray,
                    q: np.ndarray,
                    entity: np.ndarray,
                    controls: np.ndarray | None = None,
                    grid_n: int = 100,
                    q_min: float | None = None,
                    q_max: float | None = None) -> dict:
    """面板单门槛回归（个体固定效应）。

    模型：
        y_it = μ_i + β1 · x_it · I(q_it ≤ γ) + β2 · x_it · I(q_it > γ)
               + δ · controls_it + ε_it

    算法（可逐步讲解）：
        1. within 变换消除 μ_i（个体固定效应）
        2. 在门槛变量 q 的取值网格上，逐个候选 γ 做分段 OLS，记录 SSE(γ)
        3. 取使 SSE 最小的 γ* 作为门槛估计
        4. 在 γ* 下重新拟合，输出两段斜率 β1、β2

    参数：
        y        : 被解释变量（如系统效率）
        x        : 核心解释变量（如农村居民收入）
        q        : 门槛变量（可与 x 相同，也可不同）
        entity   : 实体分组标识（用于固定效应）
        controls : 控制变量矩阵 (n, k)，可空
        grid_n   : 门槛搜索网格点数
        q_min/q_max : 门槛搜索区间（默认取 q 的 5%–95% 分位，避免端点退化）

    返回字典：gamma（门槛估计）、beta1、beta2、sse、grid（搜索结果表）。
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    q = np.asarray(q, dtype=float)
    entity = np.asarray(entity)
    n = y.shape[0]

    if not (len(x) == len(q) == len(entity) == n):
        raise ValueError("y/x/q/entity 长度必须一致")

    lo = q_min if q_min is not None else np.percentile(q, 5)
    hi = q_max if q_max is not None else np.percentile(q, 95)
    if hi <= lo:
        raise ValueError("门槛搜索区间非法（hi <= lo）")
    grid = np.linspace(lo, hi, grid_n)

    # 构造设计矩阵（within 后）：[x·I(q≤γ), x·I(q>γ), controls...]
    y_w = _within_transform(y, entity)

    base_cols = [x] if controls is None else [x] + [np.asarray(c, dtype=float) for c in controls.T]
    # controls 约定为 (n, k) 矩阵；controls.T 逐列展开

    results = []
    for gamma in grid:
        d1 = (q <= gamma).astype(float)
        d2 = 1.0 - d1
        X_raw = np.column_stack([x * d1, x * d2] + base_cols[1:])
        X_w = np.apply_along_axis(lambda col: _within_transform(col, entity), 0, X_raw)
        _, sse = _ols_ssr(y_w, X_w)
        results.append((gamma, sse))

    gamma_star, sse_min = min(results, key=lambda t: t[1])

    # 在最优门槛下重新拟合，得到两段系数
    d1 = (q <= gamma_star).astype(float)
    d2 = 1.0 - d1
    X_raw = np.column_stack([x * d1, x * d2] + base_cols[1:])
    X_w = np.apply_along_axis(lambda col: _within_transform(col, entity), 0, X_raw)
    coef, _ = _ols_ssr(y_w, X_w)

    return {
        "gamma": float(gamma_star),
        "beta1": float(coef[0]),
        "beta2": float(coef[1]),
        "controls_coef": coef[2:].tolist() if coef.size > 2 else [],
        "sse": sse_min,
        "grid": pd.DataFrame(results, columns=["gamma", "sse"]),
    }


# ---------------------------------------------------------------------------
# 2. 分位数回归
# ---------------------------------------------------------------------------
def quantile_regression(y: np.ndarray,
                        X: np.ndarray,
                        quantiles: Sequence[float] = (0.25, 0.5, 0.75)) -> pd.DataFrame:
    """分位数回归：估计解释变量在不同分位点的系数。

    用于回答："这个驱动因素，在低效率组和高效率组里作用一样大吗？"
    若系数随分位点单调变化，说明存在分布异质性。

    参数：
        y         : 被解释变量
        X         : 解释变量矩阵 (n, k)，含截距列则返回对应行
        quantiles : 要估计的分位点

    返回：DataFrame，行 = 变量（含截距），列 = 各分位点系数。
    """
    from statsmodels.regression.quantile_regression import QuantReg

    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    X_c = np.column_stack([np.ones(X.shape[0]), X])     # 补截距
    col_names = ["const"] + [f"x{i}" for i in range(X.shape[1])]

    out = {}
    for tau in quantiles:
        model = QuantReg(y, X_c).fit(q=tau)
        out[tau] = model.params

    return pd.DataFrame(out, index=col_names)
