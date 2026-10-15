# -*- coding: utf-8 -*-
"""clean baseline 运行器（webshop conda 环境执行，模型经 socket 由 base 环境服务）。

用法:
    /home/ma-user/conda-envs/webshop/bin/python -m baseline.run_clean \
        --socket /tmp/aa_model.sock --goals 0,1 \
        --out data_out/baseline_debug --max-turns 15 --max-new-tokens 384

产物: {out}/manifest.json, {out}/goal_{i}.json, {out}/summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.agent_loop import run_task  # noqa: E402
from agent.config import ExperimentConfig, debug_config  # noqa: E402
from agent.env_wrapper import EnvConfig, WebShopEnvWrapper  # noqa: E402
from agent.model_client import ModelClient  # noqa: E402
from agent.schema import validate_trajectory  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", default="/tmp/aa_model.sock")
    parser.add_argument("--goals", required=True, help="逗号分隔的 goal 下标")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--human-goals", type=int, default=0)
    parser.add_argument("--num-products", type=int, default=None,
                        help="None=全量商品库；冻结任务集(task_registry)基于全量库，"
                             "正式运行不得传值")
    parser.add_argument("--max-turns", type=int, default=None,
                        help="缺省用正式配置 30；调试时显式给小值")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task-id-suffix", default="",
                        help="同一 goal 多次运行时区分文件名（确定性检查用）")
    args = parser.parse_args()

    if args.max_turns is not None or args.max_new_tokens is not None:
        cfg = debug_config(max_turns=args.max_turns or 15,
                           max_new_tokens=args.max_new_tokens or 384)
    else:
        cfg = ExperimentConfig()
    cfg.env = EnvConfig(seed=args.seed, human_goals=bool(args.human_goals),
                        num_products=args.num_products)

    wrapper = WebShopEnvWrapper(cfg.env)
    client = ModelClient(args.socket)
    if not client.ping():
        print(f"模型服务不可达: {args.socket}", file=sys.stderr)
        return 1
    print(f"[runner] 模型服务 OK; goals={args.goals}; "
          f"config_sha={cfg.manifest_sha()[:12]}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.json").write_text(
        json.dumps(cfg.manifest() | {"manifest_sha256": cfg.manifest_sha()},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    goal_ids = [int(x) for x in args.goals.split(",") if x.strip() != ""]
    summary = []
    for gi in goal_ids:
        t0 = time.time()
        traj = run_task(wrapper, client, gi, cfg)
        problems = validate_trajectory(traj)
        name = f"goal_{gi}{args.task_id_suffix}"
        traj.task_id = name
        traj.dump(args.out / f"{name}.json")
        row = {
            "task_id": name,
            "goal_idx": gi,
            "done_reason": traj.outcome.done_reason,
            "purchased_asin": traj.outcome.purchased_asin,
            "final_reward": traj.outcome.final_reward,
            "r_price": traj.outcome.r_price,
            "steps": traj.outcome.steps_used,
            "wall_s": round(time.time() - t0, 1),
            "schema_problems": problems,
        }
        summary.append(row)
        print(f"[runner] {name}: {row['done_reason']} "
              f"reward={row['final_reward']} steps={row['steps']} "
              f"({row['wall_s']}s) problems={problems or '无'}", flush=True)

    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    n_purchase = sum(1 for r in summary if r["done_reason"] == "purchase")
    print(f"[runner] 完成 {len(summary)} 任务, 购买 {n_purchase}, "
          f"输出 {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
