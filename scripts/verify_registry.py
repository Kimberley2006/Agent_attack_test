"""校验冻结任务集在当前环境可精确复现（在 webshop conda 环境中运行）。

用与冻结时完全一致的环境定义（random.seed(233) + 全量商品库 + human_goals）
重建 goal 列表，逐字段比对 task_registry.json 中三个 split 的全部任务。
任何不一致都意味着冻结任务集失效，正式运行前必须先解决。

用法:
  WEBSHOP_ITEM_PATH=... WEBSHOP_ATTR_PATH=... WEBSHOP_HUMAN_INS_PATH=... \
  python scripts/verify_registry.py [--splits debug,dev,baseline]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.env_wrapper import EnvConfig, WebShopEnvWrapper  # noqa: E402

FIELDS = ["instruction_text", "asin", "price_upper", "goal_options",
          "attributes", "query", "name"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default=str(ROOT / "task_registry.json"))
    ap.add_argument("--splits", default="debug,dev,baseline")
    args = ap.parse_args()

    reg = json.load(open(args.registry))
    cfg = EnvConfig(seed=233, human_goals=True, num_products=None)
    wrapper = WebShopEnvWrapper(cfg)
    goals = wrapper.goals
    print(f"env goals: {len(goals)} (registry n_total_goals={reg['n_total_goals']})")
    if len(goals) != reg["n_total_goals"]:
        print("goal 总数不一致，冻结任务集不可用")
        return 1

    n_ok = n_bad = 0
    for split in args.splits.split(","):
        for t in reg["splits"][split]:
            g = goals[t["goal_idx"]]
            diffs = {f: (t.get(f), g.get(f)) for f in FIELDS
                     if t.get(f) != g.get(f)}
            if diffs:
                n_bad += 1
                print(f"MISMATCH {split} goal_idx={t['goal_idx']}: {list(diffs)}")
                for f, (a, b) in list(diffs.items())[:3]:
                    print(f"  {f}: registry={a!r} env={b!r}")
            else:
                n_ok += 1
    print(f"校验: {n_ok} 一致, {n_bad} 不一致")
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
