# -*- coding: utf-8 -*-
"""clean baseline 与攻击分支共用的轨迹日志契约。

同状态配对实验的前提是分支可精确重放，因此每步必须记录：
- obs：环境返回的原始观测（送入 prompt 构造之前的文本，未做任何改写）；
- action：本步对 obs 执行的动作（重放 obs_k 需要 action_0..action_{k-1}）；
- messages：本次模型调用的完整 prompt（分支时按位替换 obs 内容用）；
- page_url：browser.current_url（识别 item_page / item_sub_page 与 asin）。

字段缺失时宁可报错也不要静默降级——重放对不上比缺数据更难排查。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

DONE_REASONS = ("purchase", "env_done", "step_limit", "error", "aborted")


@dataclass
class StepRecord:
    step: int                      # 从 0 开始；step k 的 obs 是对 action_{k-1} 的响应
    obs: str                       # 原始观测，text 模式（' [SEP] ' 拼接的文本节点）
    action: str | None             # 本步执行的动作，'search[...]' / 'click[...]'
    page_url: str | None = None    # 该观测对应页面的 URL
    messages: list = field(default_factory=list)  # 本次模型调用的完整 prompt
    raw_output: str = ""           # 模型原始生成（含思考）
    thought: str | None = None     # 实际保留进历史的思考内容
    action_valid: bool = True
    reward: float | None = None    # env.step 返回的 status['reward']
    done: bool = False             # env.step 返回的 status['done']
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StepRecord":
        return cls(**d)


@dataclass
class OutcomeRecord:
    done_reason: str               # DONE_REASONS 之一
    purchased_asin: str | None = None
    purchased_options: dict = field(default_factory=dict)
    purchased_price: float | None = None   # env 侧 product_prices[asin]
    final_reward: float | None = None      # WebShop get_reward 总分
    r_price: bool | None = None            # price <= goal.price_upper；未购买为 None
    steps_used: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "OutcomeRecord":
        return cls(**d)


@dataclass
class TaskTrajectory:
    task_id: str
    session: str                   # WebShop session id（重放时重建）
    instruction_text: str
    goal: dict                     # WebShop goal 原始 dict（含 price_upper/attributes/goal_options）
    config_ref: str                # 冻结配置 manifest 的哈希，用于运行前一致性比对
    steps: list = field(default_factory=list)
    outcome: OutcomeRecord | None = None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "session": self.session,
            "instruction_text": self.instruction_text,
            "goal": self.goal,
            "config_ref": self.config_ref,
            "steps": [s.to_dict() if isinstance(s, StepRecord) else s for s in self.steps],
            "outcome": self.outcome.to_dict() if self.outcome is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TaskTrajectory":
        steps = [StepRecord.from_dict(s) for s in d.get("steps", [])]
        outcome = d.get("outcome")
        return cls(
            task_id=d["task_id"],
            session=d["session"],
            instruction_text=d["instruction_text"],
            goal=d.get("goal", {}),
            config_ref=d.get("config_ref", ""),
            steps=steps,
            outcome=OutcomeRecord.from_dict(outcome) if outcome else None,
        )

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "TaskTrajectory":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def validate_trajectory(traj: TaskTrajectory) -> list[str]:
    """轻量结构校验，返回问题列表（空列表 = 通过）。"""
    problems = []
    if not traj.task_id:
        problems.append("缺少 task_id")
    if not traj.instruction_text:
        problems.append("缺少 instruction_text")
    if "price_upper" not in traj.goal:
        problems.append("goal 缺少 price_upper（无法机器判定预算约束）")
    for i, s in enumerate(traj.steps):
        if s.step != i:
            problems.append(f"step 编号不连续: 位置 {i} 是 step {s.step}")
        if not s.obs or not s.obs.strip():
            problems.append(f"step {i} obs 为空")
        if i < len(traj.steps) - 1 and not s.action:
            problems.append(f"step {i} 缺少 action（中间步必须有动作才能重放）")
    if traj.outcome is not None and traj.outcome.done_reason not in DONE_REASONS:
        problems.append(f"未知 done_reason: {traj.outcome.done_reason}")
    return problems
