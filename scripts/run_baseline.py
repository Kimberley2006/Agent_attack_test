"""Clean baseline 主入口（base 环境，需先启动 env_service）。

用法：
  CUDA_VISIBLE_DEVICES=1 python scripts/run_baseline.py \
      --split debug [--collect basic|off] [--limit N] [--tag xxx]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_attack.config import load_config, make_run_dir, snapshot_code  # noqa: E402
from agent_attack.env_client import EnvClient  # noqa: E402
from agent_attack.episode_logger import EpisodeLogger  # noqa: E402
from agent_attack.model_backend import ModelBackend  # noqa: E402


def load_inject_spec(path):
    """注入实验规格 v0：
    {"name": ..., "trigger": {"page_type": "item_page", "once": true},
     "payload": "追加到观测末尾的文本"}
    注入发生在页面已生成、尚未交给模型决策的时刻（方案⑨）。
    返回 (spec, make_transform)：make_transform() 每个任务生成全新 transform，
    once/状态按任务隔离，避免跨任务泄漏。"""
    spec = json.loads(Path(path).read_text())
    assert "payload" in spec and "trigger" in spec

    def make_transform():
        state = {"done": False}

        def transform(obs, page_type, step):
            trig = spec["trigger"]
            if state["done"] and trig.get("once", True):
                return obs, None
            if trig.get("page_type") and page_type != trig["page_type"]:
                return obs, None
            if "at_step" in trig and step != trig["at_step"]:
                return obs, None
            state["done"] = True
            return obs.rstrip() + "\n\n" + spec["payload"] + "\n", spec["payload"]

        return transform

    return spec, make_transform


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="debug",
                    choices=["debug", "dev", "baseline", "all"])
    ap.add_argument("--collect", default=None,
                    choices=[None, "off", "basic", "detailed"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--task-ids", type=str, default=None,
                    help="逗号分隔 goal_idx，覆盖 split（分片/重跑用）")
    ap.add_argument("--service-url", default=None,
                    help="覆盖 config 中的环境服务地址（并行分片用）")
    ap.add_argument("--inject-file", default=None,
                    help="注入规格 JSON（攻击实验）；结果永不与 clean 混合统计")
    args = ap.parse_args()

    cfg = load_config(args.config)
    collect = args.collect if args.collect is not None \
        else cfg.get("representation.collect")
    service_url = args.service_url or cfg.get("environment.service_url")

    inject_spec, make_transform = (None, None)
    if args.inject_file:
        inject_spec, make_transform = load_inject_spec(args.inject_file)

    registry = json.loads(Path(cfg.get("tasks.registry")).read_text())
    if args.task_ids:
        by_idx = {t["goal_idx"]: t for s in registry["splits"].values()
                  for t in s}
        tasks = [by_idx[int(i)] for i in args.task_ids.split(",")]
    else:
        splits = registry["splits"]
        tasks = splits[args.split] if args.split != "all" else \
            [t for s in splits.values() for t in s]
    if args.limit:
        tasks = tasks[:args.limit]

    run_dir = make_run_dir(cfg, args.tag or args.split)
    snapshot_code(run_dir, cfg)
    run_id = run_dir.name

    env = EnvClient(service_url)
    service_info = env.health()
    print(f"service: {service_info}")

    logger = EpisodeLogger(run_dir, cfg, service_info,
                           {"registry_created": registry["created_utc"],
                            "split": args.split, "n_tasks": len(tasks),
                            "data_fingerprint": registry["data_fingerprint"],
                            "inject_spec": inject_spec})

    print(f"loading model (collect={collect}) ...")
    backend = ModelBackend(cfg, collect=collect)
    print(f"run_dir: {run_dir}")

    for i, task in enumerate(tasks):
        t0 = time.time()
        from agent_attack.agent import run_episode
        obs_transform = make_transform() if make_transform else None
        rec = run_episode(task, backend, env, logger, run_id, cfg,
                          collect_reps=(collect != "off"),
                          obs_transform=obs_transform)
        out = rec["outcome"]
        inj = f" inj={rec['injections']['n']}" if rec["injections"]["n"] else ""
        print(f"[{i+1}/{len(tasks)}] task {task['goal_idx']}: "
              f"{out['status']} reward={out['reward']} "
              f"steps={rec['n_model_calls']} "
              f"inv={rec['n_invalid_actions']} parse_fail={rec['n_parse_failures']}"
              f"{inj} ({time.time()-t0:.0f}s)", flush=True)

    print("done. metrics:")
    import subprocess
    subprocess.run([sys.executable, str(Path(__file__).parent / "compute_metrics.py"),
                    str(run_dir)], check=False)


if __name__ == "__main__":
    main()
