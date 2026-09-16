# -*- coding: utf-8 -*-
"""quickstart.py —— 端到端演示（可替代 Notebook 直接运行）。

用法：
    python examples/quickstart.py

它会：生成模拟数据 -> 跑完整效率评估流程 -> 打印关键结果 -> 输出图表到 output/。
"""

import sys
from pathlib import Path

# 让脚本能 import 到 src 下的包
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dea_toolkit import data, pipeline  # noqa: E402


def main():
    # 1. 生成模拟面板数据（31 实体 × 11 年）
    df = data.make_demo_data(n_entity=31, n_year=11, seed=42)
    print("数据形状:", df.shape)
    print(df.head(3).to_string(index=False), "\n")

    # 2. 一键端到端
    res = pipeline.run_end_to_end(
        df,
        stage1_inputs=["人力资本", "财政支农"],
        stage1_outputs=["数字基建"],
        stage2_outputs=["创业活跃度"],
        x_var="收入",
        q_var="收入",
        output_dir="output",
        frontier="yearly",
    )

    eff_df = res["eff_df"]
    print("===== 两阶段 DEA 效率 =====")
    print("阶段一效率均值:", round(eff_df["stage1_eff"].mean(), 4))
    print("阶段二效率均值:", round(eff_df["stage2_eff"].mean(), 4))
    print("系统效率均值  :", round(eff_df["system_eff"].mean(), 4), "\n")

    print("===== 门槛回归 =====")
    th = res["threshold"]
    print("门槛估计值 gamma:", round(th["gamma"], 4))
    print("低区间斜率 beta1:", round(th["beta1"], 4))
    print("高区间斜率 beta2:", round(th["beta2"], 4), "\n")

    print("===== 分位数回归（收入 -> 系统效率）=====")
    print(res["quantile"].round(4), "\n")

    print("===== 马尔科夫转移矩阵 =====")
    print(res["markov"], "\n")

    print("===== 图表已保存 =====")
    for f in res["figs"]:
        print(" -", f)


if __name__ == "__main__":
    main()
