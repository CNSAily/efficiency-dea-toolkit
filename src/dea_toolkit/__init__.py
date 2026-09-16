# -*- coding: utf-8 -*-
"""dea_toolkit —— 经管效率评估工具包。

端到端流程：
    原始投入产出面板数据
        -> data.normalize_panel()          数据预处理
        -> dea.two_stage_eff()             两阶段 DEA 效率测算
        -> mechanism.threshold_reg()       门槛效应 / 分位数回归
        -> visualize.*                     可视化
        -> pipeline.run_end_to_end()       一键报告

版本与作者信息见 README。
"""

__version__ = "0.1.0"
__author__ = "王丹"

from . import data
from . import dea
from . import mechanism
from . import visualize

__all__ = ["data", "dea", "mechanism", "visualize", "__version__"]
