# -*- coding: utf-8 -*-
"""dea.py —— 自研 DEA（数据包络分析）求解器。

核心价值：不依赖任何"黑盒 DEA 软件"（MaxDEA / DEAP 等），
直接用 scipy.optimize.linprog 实现 DEA 的线性规划，把"DEA 本质是
线性规划"这件事讲清楚、算明白。这是本工具包含金量最高的部分。

实现的模型：
    - CCR（Charnes-Cooper-Rhodes, 1978）：规模报酬不变（CRS）
    - BCC（Banker-Charnes-Cooper, 1984）：规模报酬可变（VRS）
    - 两阶段 DEA：分别测算"阶段一效率 E1"与"阶段二效率 E2"，
      系统效率 = E1 × E2（可解释为两阶段综合传导）

DEA 直觉（面试可讲）：
    给定一批决策单元（DMU），每个单元有若干投入 X 与产出 Y。
    我们问：在"投入不超过某单元、产出不低于某单元"的前提下，
    能否通过给各单元加权组合（λ）构造出一个"虚拟单元"，它在投入
    相同（或更少）时产出比该单元还高？若能，说明该单元相对低效。
    这个"最多能扩张多少产出 / 最少能压缩多少投入"的比例，就是效率。
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
from scipy.optimize import linprog


# ---------------------------------------------------------------------------
# 1. 单 DMU 线性规划求解器（包络形式）
# ---------------------------------------------------------------------------
def _solve_dmu(X: np.ndarray, Y: np.ndarray, k: int,
               orientation: str = "output",
               vrs: bool = False) -> float:
    """求解第 k 个 DMU 的 DEA 效率（包络形式）。

    参数：
        X : (N, m) 投入矩阵，第 k 行为待评估 DMU 的投入
        Y : (N, s) 产出矩阵
        k : 待评估 DMU 下标
        orientation : 'output' 产出导向 / 'input' 投入导向
        vrs : True 为 BCC（VRS，加 Σλ=1），False 为 CCR（CRS）

    返回效率得分 ∈ (0, 1]，1 表示 DEA 有效。
    """
    N, m = X.shape
    s = Y.shape[1]
    x_k = X[k]                 # (m,)
    y_k = Y[k]                 # (s,)

    if orientation == "output":
        # max φ  s.t. Yλ ≥ φ·y_k ; Xλ ≤ x_k ; λ ≥ 0
        # 等价于 min -φ
        n_var = N + 1                                   # [λ_1..λ_N, φ]
        c = np.zeros(n_var)
        c[-1] = -1.0

        A_ub = np.zeros((s + m, n_var))
        b_ub = np.zeros(s + m)
        # 产出约束：-(Yλ) + φ·y_k ≤ 0   =>   A[:s, :N] = -Y, A[:s, -1] = y_k
        A_ub[:s, :N] = -Y.T
        A_ub[:s, -1] = y_k
        # 投入约束：Xλ ≤ x_k            =>   A[s:, :N] = X
        A_ub[s:, :N] = X.T
        b_ub[s:] = x_k

        bounds = [(0, None)] * N + [(None, None)]        # λ≥0，φ 自由

        # VRS（BCC）：附加凸性约束 Σλ = 1
        if vrs:
            A_eq = np.ones((1, n_var))
            A_eq[0, -1] = 0.0
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                          bounds=bounds, method="highs")
        else:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

        if not res.success:
            raise RuntimeError(f"DMU {k} 线性规划未收敛：{res.message}")
        phi = res.x[-1]
        if phi <= 0:
            return 0.0
        return min(1.0, 1.0 / phi)

    elif orientation == "input":
        # min θ  s.t. Yλ ≥ y_k ; Xλ ≤ θ·x_k ; λ ≥ 0
        n_var = N + 1                                   # [λ_1..λ_N, θ]
        c = np.zeros(n_var)
        c[-1] = 1.0

        A_ub = np.zeros((s + m, n_var))
        b_ub = np.zeros(s + m)
        # 产出约束：Yλ ≥ y_k  =>  -Yλ ≤ -y_k
        A_ub[:s, :N] = -Y.T
        b_ub[:s] = -y_k
        # 投入约束：Xλ - θ·x_k ≤ 0
        A_ub[s:, :N] = X.T
        A_ub[s:, -1] = -x_k

        bounds = [(0, None)] * N + [(None, None)]

        # VRS（BCC）：附加凸性约束 Σλ = 1
        if vrs:
            A_eq = np.ones((1, n_var))
            A_eq[0, -1] = 0.0
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                          bounds=bounds, method="highs")
        else:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

        if not res.success:
            raise RuntimeError(f"DMU {k} 线性规划未收敛：{res.message}")
        return min(1.0, max(0.0, res.x[-1]))

    else:
        raise ValueError("orientation 只能是 'output' 或 'input'")


# ---------------------------------------------------------------------------
# 2. CCR / BCC 公开接口
# ---------------------------------------------------------------------------
def dea_efficiency(X: np.ndarray, Y: np.ndarray,
                   orientation: str = "output",
                   returns: Literal["crs", "vrs"] = "crs") -> np.ndarray:
    """计算一组 DMU 的 DEA 效率。

    参数：
        X : (N, m) 投入矩阵
        Y : (N, s) 产出矩阵
        orientation : 'output' / 'input'
        returns : 'crs'（CCR，规模报酬不变）/'vrs'（BCC，规模报酬可变）

    返回：效率数组，长度 N，∈ (0, 1]。
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
        raise ValueError("X、Y 必须为二维矩阵且行数（DMU 数）一致")
    if np.any(X <= 0) or np.any(Y <= 0):
        raise ValueError("DEA 要求投入/产出严格为正")

    vrs = returns == "vrs"
    eff = [_solve_dmu(X, Y, k, orientation=orientation, vrs=vrs)
           for k in range(X.shape[0])]
    return np.array(eff)


def ccr(X: np.ndarray, Y: np.ndarray, orientation: str = "output") -> np.ndarray:
    """CCR 模型（CRS）。"""
    return dea_efficiency(X, Y, orientation=orientation, returns="crs")


def bcc(X: np.ndarray, Y: np.ndarray, orientation: str = "output") -> np.ndarray:
    """BCC 模型（VRS）。"""
    return dea_efficiency(X, Y, orientation=orientation, returns="vrs")


# ---------------------------------------------------------------------------
# 3. 两阶段 DEA
# ---------------------------------------------------------------------------
def two_stage_eff(X1: np.ndarray, Z: np.ndarray, Y2: np.ndarray,
                  orientation: str = "output",
                  returns: Literal["crs", "vrs"] = "crs") -> dict[str, np.ndarray]:
    """两阶段 DEA：分别测算阶段一、阶段二效率，并合成系统效率。

    结构：
        阶段一：X1（资源禀赋投入） -> Z（中间产出）
        阶段二：Z（作为投入）       -> Y2（最终产出）
        系统效率 = 阶段一效率 × 阶段二效率

    返回字典：{'stage1': E1, 'stage2': E2, 'system': E1*E2}
    """
    E1 = dea_efficiency(X1, Z, orientation=orientation, returns=returns)
    E2 = dea_efficiency(Z, Y2, orientation=orientation, returns=returns)
    return {"stage1": E1, "stage2": E2, "system": E1 * E2}
