# -*- coding: utf-8 -*-
"""ReAct agent 循环：驱动 WebShop 任务并按 schema 记录完整轨迹。

设计约束（配对实验的前提）：
- prompt 构造是纯函数：messages = f(system, instruction, [(obs, assistant)])，
  分支运行器用同样的函数从轨迹重建任意步的模型输入；
- 模型是 thinking 模型：raw_output 含 <think> 块；历史中只保留思考后的
  最终可见内容（含 Thought/Action 两行），完整 raw_output 记入轨迹；
- 动作解析失败给一次纠错机会（追加纠正性 user 消息），仍失败则终止任务
  （done_reason=error）；连续无效点击（观测不变）达到上限同样终止；
- 终局信息（购买 asin/价格/r_price/reward）全部来自 env 真实状态。
"""
from __future__ import annotations

import re

from .config import ExperimentConfig
from .env_wrapper import WebShopEnvWrapper
from .model_client import ModelClient
from .schema import OutcomeRecord, StepRecord, TaskTrajectory

_THINK_SPLIT = re.compile(r"</think>", re.IGNORECASE)
_ACTION_LINE = re.compile(r"Action:\s*(search|click)\[([^\[\]]*)\]", re.IGNORECASE)
_DONE_ASIN_RE = re.compile(r"/done/[^/]+/([A-Z0-9]{10})")


def parse_generation(raw: str) -> tuple[str, str]:
    """拆分 thinking 与最终可见内容。无 </think> 时整体视为可见内容。"""
    parts = _THINK_SPLIT.split(raw, maxsplit=1)
    if len(parts) == 2:
        thinking = parts[0].replace("<think>", "", 1).strip()
        return thinking, parts[1].strip()
    return "", raw.strip()


def extract_action(content: str) -> str | None:
    """取最后一条合法 Action 行，归一化为小写参数的 search[...]/click[...]。"""
    matches = _ACTION_LINE.findall(content)
    if not matches:
        return None
    name, arg = matches[-1]
    arg = arg.strip()
    return f"{name.lower()}[{arg}]"


def initial_messages(system_prompt: str, instruction: str, obs: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Task: {instruction}\n\nObservation:\n{obs}"},
    ]


def append_turn(messages: list[dict], assistant_content: str, obs: str) -> None:
    messages.append({"role": "assistant", "content": assistant_content})
    messages.append({"role": "user", "content": f"Observation:\n{obs}"})


def _action_valid(wrapper: WebShopEnvWrapper, action: str) -> bool:
    name, _, arg = action.partition("[")
    arg = arg.rstrip("]").lower()
    if name == "search":
        return bool(arg)
    if name == "click":
        return arg in wrapper.available_actions()
    return False


def run_task(wrapper: WebShopEnvWrapper, client: ModelClient, goal_idx: int,
             cfg: ExperimentConfig, meta_prefix: str | None = None) -> TaskTrajectory:
    goal = wrapper.goal(goal_idx)
    instruction = goal["instruction_text"]
    task_id = f"goal_{goal_idx}"
    gen_meta = (lambda turn: {"task_id": meta_prefix or task_id,
                              "step": turn})

    first = wrapper.reset_fresh(goal_idx)
    obs, url = first.obs, first.url
    messages = initial_messages(cfg.system_prompt, instruction, obs)

    steps: list[StepRecord] = []
    done_reason, outcome = "step_limit", None
    no_progress = 0

    for turn in range(cfg.max_turns):
        step = StepRecord(step=turn, obs=obs, action=None, page_url=url,
                          messages=[dict(m) for m in messages])
        retries: list[dict] = []

        action, content, raw, resp = None, None, None, None
        for attempt in range(cfg.max_model_calls_per_turn):
            resp = client.generate(
                messages,
                max_new_tokens=cfg.gen.max_new_tokens_per_turn,
                do_sample=cfg.gen.do_sample,
                template_kwargs={"enable_thinking": cfg.gen.enable_thinking},
                meta=gen_meta(turn),
            )
            raw = resp["output_text"]
            _, content = parse_generation(raw)
            action = extract_action(content)
            if action is not None:
                break
            retries.append({
                "raw_output": raw,
                "n_new_tokens": resp.get("n_new_tokens"),
                "truncated": resp.get("truncated"),
            })
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": "Your reply contained no valid 'Action:' line. "
                           "Reply again with a short Thought and exactly one "
                           "Action line (search[...] or click[...]).",
            })

        if action is None:
            step.raw_output = raw or ""
            step.thought = content
            step.action_valid = False
            step.extra = {"retries": retries, "abort": "unparseable_action",
                          "n_prompt_tokens": resp.get("n_prompt_tokens") if resp else None}
            steps.append(step)
            done_reason = "error"
            break

        valid = _action_valid(wrapper, action)
        step.action = action
        step.action_valid = valid
        step.raw_output = raw
        step.thought = content
        step.reward = None
        step.extra = {
            "retries": retries,
            "n_prompt_tokens": resp.get("n_prompt_tokens"),
            "n_new_tokens": resp.get("n_new_tokens"),
            "truncated": resp.get("truncated"),
            "call_index": resp.get("call_index"),
        }
        steps.append(step)

        result = wrapper.step(action)
        step.reward = result.reward
        step.done = result.done

        if result.obs == obs and not result.done:
            no_progress += 1
        else:
            no_progress = 0
        if no_progress >= cfg.noop_abort:
            done_reason = "aborted"
            break

        # 历史追加：失败尝试与纠错消息已在 messages 中（模型确实看过），
        # 这里追加成功回复与新观测
        append_turn(messages, content, result.obs)
        obs, url = result.obs, result.url

        if result.done:
            done_reason = "purchase"
            m = _DONE_ASIN_RE.search(url)
            asin = m.group(1) if m else None
            price = wrapper.price(asin) if asin else None
            price_upper = goal.get("price_upper")
            has_constraint = price_upper is not None and 0 < price_upper < 1000000
            outcome = OutcomeRecord(
                done_reason="purchase",
                purchased_asin=asin,
                purchased_price=price,
                final_reward=result.reward,
                r_price=(price <= price_upper) if (asin and has_constraint) else None,
                steps_used=turn + 1,
            )
            break

    if outcome is None:
        outcome = OutcomeRecord(done_reason=done_reason, steps_used=len(steps))

    return TaskTrajectory(
        task_id=task_id,
        session=wrapper._session_key(goal_idx),
        instruction_text=instruction,
        goal=goal,
        config_ref=cfg.manifest_sha(),
        steps=steps,
        outcome=outcome,
    )
