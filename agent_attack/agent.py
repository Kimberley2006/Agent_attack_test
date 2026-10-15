"""ReAct 决策循环（方案③）。不替模型做任何选择：不自动选商品、不修正参数、
不补购买步骤；解析失败/无效动作/预算耗尽全部进日志。"""

import time
from typing import Optional

from agent_attack import prompts
from agent_attack.action_parser import parse_action


class EpisodeStats:
    def __init__(self):
        self.n_model_calls = 0
        self.n_actions_executed = 0
        self.n_invalid_actions = 0
        self.n_parse_failures = 0
        self.n_prompt_tokens_total = 0
        self.n_output_tokens_total = 0
        self.wall_seconds = 0.0
        self.visited_page_types = set()

    def as_dict(self):
        return {
            "n_model_calls": self.n_model_calls,
            "n_actions_executed": self.n_actions_executed,
            "n_invalid_actions": self.n_invalid_actions,
            "n_parse_failures": self.n_parse_failures,
            "n_prompt_tokens_total": self.n_prompt_tokens_total,
            "n_output_tokens_total": self.n_output_tokens_total,
            "wall_seconds": self.wall_seconds,
            "visited_page_types": sorted(self.visited_page_types),
        }


def run_episode(task, backend, env, logger, run_id, cfg,
                collect_reps=False, obs_override=None, obs_transform=None):
    """执行一个任务。task: registry 条目（含 goal_idx 与 instruction_text）。

    obs_override: 用于重放/注入实验——替换初始观测（保持其余流程一致）。
    obs_transform: fn(obs_text, page_type, step) -> (new_obs, injected_payload|None)
    ——注入实验的统一入口：在页面已生成、尚未交给模型决策的时刻改写观测。
    改写只影响模型输入；环境侧始终执行真实动作，日志同时保留原始转移。
    """
    task_id = task["goal_idx"]
    stats = EpisodeStats()
    stats.n_injections = 0
    stats.injected_steps = []
    t0 = time.time()

    reset = env.reset(task_id)
    instruction = task["instruction_text"]
    if reset["instruction_text"].strip() != instruction.strip():
        raise RuntimeError(
            f"task {task_id} instruction drift:\n registry: {instruction!r}\n "
            f"env: {reset['instruction_text']!r}")
    obs = obs_override if obs_override is not None else reset["obs"]
    page_type = reset["page_type"]
    injected_payload = None
    if obs_transform is not None:
        obs, injected_payload = obs_transform(obs, page_type, 0)
        if injected_payload:
            stats.n_injections += 1
            stats.injected_steps.append(0)

    messages = [
        {"role": "system", "content": prompts.SYSTEM_PROMPT},
        {"role": "user", "content": prompts.build_task_message(instruction)
         + "\n\n" + prompts.format_observation(obs, first=True)},
    ]

    max_turns = cfg.get("agent.max_turns")
    parse_feedback = cfg.get("agent.parse_error_feedback")
    outcome = {"status": "budget_exhausted", "purchased": None,
               "reward": None, "done": False}
    done = False

    for step in range(max_turns):
        model_call_id = f"t{task_id}_s{step}"
        res, cap = backend.generate(messages, model_call_id)
        stats.n_model_calls += 1
        stats.n_prompt_tokens_total += res.n_prompt_tokens
        stats.n_output_tokens_total += res.n_output_tokens
        stats.wall_seconds = time.time() - t0

        action, parse_info = parse_action(res.output_text)
        action_sub = action
        if action is None:
            stats.n_parse_failures += 1
        backend.finalize(res, cap, action_sub)

        tensor_file = None
        if collect_reps and res.reps_payload is not None:
            tensor_file = logger.save_tensor(res.reps_payload, model_call_id)

        if action is None:
            # 固定错误反馈，计入决策预算（方案③）
            logger.log_model_call(
                run_id=run_id, task_id=task_id, agent_step=step,
                model_call_id=model_call_id, messages=messages,
                prompt_ids=res.prompt_ids, output_ids=res.output_ids,
                output_text=res.output_text, finish_reason=res.finish_reason,
                think_span=res.think_span, action_span=res.action_span,
                anchors=res.anchors, gen_seconds=res.gen_seconds,
                action_str=None, parse_info=parse_info,
                action_invalid=None, transition=None, tensor_file=tensor_file)
            messages.append({"role": "assistant", "content": res.output_text})
            messages.append({"role": "user", "content": parse_feedback.strip()})
            continue

        transition = env.step(task_id, action)
        stats.visited_page_types.add(transition["page_type"])
        if transition["invalid"]:
            stats.n_invalid_actions += 1
        else:
            stats.n_actions_executed += 1

        logger.log_model_call(
            run_id=run_id, task_id=task_id, agent_step=step,
            model_call_id=model_call_id, messages=messages,
            prompt_ids=res.prompt_ids, output_ids=res.output_ids,
            output_text=res.output_text, finish_reason=res.finish_reason,
            think_span=res.think_span, action_span=res.action_span,
            anchors=res.anchors, gen_seconds=res.gen_seconds,
            action_str=action, parse_info=parse_info,
            action_invalid=transition["invalid"],
            transition={k: transition[k] for k in
                        ("action", "invalid", "reward", "done", "obs", "url",
                         "page_type", "available_actions", "purchased")},
            tensor_file=tensor_file)

        messages.append({"role": "assistant", "content": res.output_text})
        next_obs = transition["obs"]
        if obs_transform is not None:
            next_obs, injected_payload = obs_transform(
                next_obs, transition["page_type"], step + 1)
            if injected_payload:
                stats.n_injections += 1
                stats.injected_steps.append(step + 1)
                transition = dict(transition)
                transition["injected_payload"] = injected_payload
        messages.append({"role": "user", "content":
                         prompts.format_observation(next_obs)})

        if transition["done"]:
            outcome = {
                "status": "purchased",
                "purchased": transition["purchased"],
                "reward": transition["purchased"]["reward"]
                          if transition["purchased"] else None,
                "verbose_info": transition["purchased"]["verbose_info"]
                                if transition["purchased"] else None,
                "done": True,
            }
            done = True
            break

    if not done:
        if stats.n_parse_failures == stats.n_model_calls and stats.n_model_calls > 0:
            outcome = {"status": "all_parse_failures", "purchased": None,
                       "reward": None, "done": False}
        else:
            outcome = {"status": "budget_exhausted", "purchased": None,
                       "reward": None, "done": False}

    env.close_session(task_id)
    stats.wall_seconds = time.time() - t0
    return logger.log_episode(
        run_id=run_id, task_id=task_id, instruction_text=instruction,
        goal_truth={k: task[k] for k in ("asin", "attributes", "goal_options",
                                         "price_upper", "query", "name")},
        steps=stats.as_dict(), outcome=outcome,
        injections={"n": stats.n_injections, "steps": stats.injected_steps})
