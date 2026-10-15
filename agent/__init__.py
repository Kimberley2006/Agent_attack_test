# -*- coding: utf-8 -*-
"""agent 包：clean baseline 与攻击分支共用的内核（配置、环境包装、循环、契约）。

模块规划：
- schema : 轨迹日志契约（已完成）
- config / env_wrapper / agent_loop / model_client : 待实现
"""
from .schema import (
    DONE_REASONS,
    OutcomeRecord,
    StepRecord,
    TaskTrajectory,
    validate_trajectory,
)

__all__ = [
    "DONE_REASONS",
    "OutcomeRecord",
    "StepRecord",
    "TaskTrajectory",
    "validate_trajectory",
]
