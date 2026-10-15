"""结构化日志（方案⑦）：每次模型调用一条 JSONL，每任务一份 episode 摘要，
表征张量单独存 .pt 并在 JSONL 中登记引用。"""

import hashlib
import json
import os
import time
from pathlib import Path


def sha16(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


class EpisodeLogger:
    def __init__(self, run_dir: Path, cfg, service_info: dict, task_registry_meta: dict):
        self.run_dir = Path(run_dir)
        self.calls_path = self.run_dir / "model_calls.jsonl"
        self.episodes_path = self.run_dir / "episodes.jsonl"
        self.tensor_dir = self.run_dir / "tensors"
        self.tensor_dir.mkdir(parents=True, exist_ok=True)
        self.header = {
            "log_version": cfg.get("logging.fields_version"),
            "prompt_version": cfg.get("agent.prompt_version"),
            "service_info": service_info,
            "task_registry_meta": task_registry_meta,
        }
        with open(self.run_dir / "log_header.json", "w") as f:
            json.dump(self.header, f, indent=2, ensure_ascii=False)

    # ---------------- 模型调用级 ----------------
    def log_model_call(self, *, run_id, task_id, agent_step, model_call_id,
                       messages, prompt_ids, output_ids, output_text,
                       finish_reason, think_span, action_span, anchors,
                       gen_seconds, action_str, parse_info, action_invalid,
                       transition, tensor_file=None):
        rec = {
            "run_id": run_id,
            "task_id": task_id,
            "agent_step": agent_step,
            "model_call_id": model_call_id,
            "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            # 输入
            "messages": messages,                       # 完整消息历史（save_full_history）
            "n_prompt_tokens": len(prompt_ids),
            "prompt_ids": prompt_ids,
            "truncated": False,                          # 第一版不截断
            # 输出
            "output_ids": output_ids,
            "output_text": output_text,
            "finish_reason": finish_reason,
            "think_span": think_span,
            "action_span": action_span,
            "gen_seconds": round(gen_seconds, 3),
            # 动作解析
            "action": action_str,
            "parse_info": parse_info,
            # 环境
            "action_invalid": action_invalid,
            "env": transition,
            # 表征
            "representation": {
                "collect": bool(tensor_file),
                "anchors_gen_pos": anchors,
                "tensor_file": str(tensor_file) if tensor_file else None,
            },
        }
        with open(self.calls_path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    def save_tensor(self, payload: dict, model_call_id: str) -> str:
        import torch
        path = self.tensor_dir / f"{model_call_id}.pt"
        torch.save(payload, path)
        return str(path)

    # ---------------- 任务级 ----------------
    def log_episode(self, *, run_id, task_id, instruction_text, goal_truth,
                    steps, outcome, injections=None):
        rec = {
            "run_id": run_id,
            "task_id": task_id,
            "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "instruction_text": instruction_text,
            "goal_truth": goal_truth,      # 评估端保存的真实约束（不进模型输入）
            "n_model_calls": steps["n_model_calls"],
            "n_actions_executed": steps["n_actions_executed"],
            "n_invalid_actions": steps["n_invalid_actions"],
            "n_parse_failures": steps["n_parse_failures"],
            "n_prompt_tokens_total": steps["n_prompt_tokens_total"],
            "n_output_tokens_total": steps["n_output_tokens_total"],
            "wall_seconds": round(steps["wall_seconds"], 2),
            "visited_page_types": steps["visited_page_types"],
            "visited_description_page": "item_sub_page_description"
                                        in steps["visited_page_types"],
            "injections": injections or {"n": 0, "steps": []},
            "outcome": outcome,            # purchased / reward / verbose_info / 结束原因
        }
        with open(self.episodes_path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec
