# efficiency-dea-toolkit · 经管效率评估工具包

面向经管研究者的**端到端效率评估工具**：输入原始投入产出面板数据，一键跑通
「指标体系 → 两阶段 DEA 效率测算 → 门槛/分位数机制分析 → 可视化报告」。

不依赖任何黑盒 DEA 软件（MaxDEA / DEAP 等），核心的 DEA 线性规划与面板门槛
算法**全部手写实现**，既可用于研究复现，也可作为学习 DEA 方法论的参考实现。

---

## ✨ 特性

- **自研 DEA 求解器**：用 `scipy.optimize.linprog` 实现 CCR（规模报酬不变）、
  BCC（规模报酬可变）与两阶段 DEA，支持投入/产出导向。
- **两阶段网络结构**：资源禀赋 → 中间产出 → 最终产出，输出阶段一、阶段二与
  系统效率三层结果。
- **机制分析**：面板门槛回归（Hansen 1999）+ 分位数回归，量化驱动因素的非线性
  与分布异质性。
- **四类可视化**：时序趋势、核密度演化、区域对比、马尔科夫转移矩阵。
- **一键端到端**：`pipeline.run_end_to_end()` 一条命令产出全部结果与图表。

## 📦 安装

```bash
pip install -r requirements.txt
```

依赖：`numpy`、`pandas`、`scipy`、`statsmodels`、`matplotlib`（均为标准科学计算栈）。

## 🚀 快速开始

```python
import sys
sys.path.insert(0, "src")

from dea_toolkit import data, pipeline

# 生成模拟演示数据（31 实体 × 11 年）
df = data.make_demo_data(n_entity=31, n_year=11, seed=42)

# 一键端到端：两阶段 DEA + 门槛/分位数 + 可视化
res = pipeline.run_end_to_end(
    df,
    stage1_inputs=["人力资本", "财政支农"],   # 第一阶段投入
    stage1_outputs=["数字基建"],              # 中间产出（兼作第二阶段投入）
    stage2_outputs=["创业活跃度"],            # 第二阶段产出
    x_var="收入",                             # 核心解释变量
    q_var="收入",                             # 门槛变量
    output_dir="output",
    frontier="yearly",                        # 逐年前沿（效率值更合理）；"global" 为全局前沿
)

print("系统效率均值:", res["eff_df"]["system_eff"].mean())
print("门槛估计值:", res["threshold"]["gamma"])
print("两段斜率 beta1/beta2:", res["threshold"]["beta1"], res["threshold"]["beta2"])
```

更多示例见 `examples/quickstart.py`。

> **前沿选择**：`frontier="yearly"` 按年份分别构造前沿（效率值落在更合理区间，适合看趋势）；
> `frontier="global"` 将所有年份混合构造前沿（可跨期直接比较，但效率绝对值偏低）。
> 面板 DEA 研究中两者均有采用，按研究目的选择。

## 🧱 模块结构

| 模块 | 职责 |
|---|---|
| `data.py` | 面板读取、非负校验、缺失插补、归一化、两阶段数据装配、演示数据生成 |
| `dea.py` | CCR / BCC / 两阶段 DEA 求解器（自研线性规划） |
| `mechanism.py` | 面板门槛回归（within 变换 + 网格搜索）、分位数回归 |
| `visualize.py` | 时序 / 核密度 / 区域对比 / 马尔科夫转移可视化 |
| `pipeline.py` | 端到端流程编排 |

## 📚 方法论

DEA 与门槛回归的完整推导见 [`docs/methodology.md`](docs/methodology.md)。

## ⚠️ 说明

- 仓库内示例数据为**模拟数据**，仅用于演示 API 与复现流程，不包含任何真实研究
  数据或结论。
- 两阶段 DEA 当前采用"两阶段独立测算 + 系统效率相乘"的经典简化口径；如需网络
  DEA 的乘积/加性分解等进阶口径，可在 `dea.py` 基础上扩展。

## 📄 License

[MIT](LICENSE)
