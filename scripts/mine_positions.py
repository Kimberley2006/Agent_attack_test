"""从 clean 轨迹挖掘注入位置（webshop conda 环境）。

与 attack/positions.py CLI 的区别：商品价格表取自实际构建的 env 实例
（wrapper.product_table()），而不是 build_product_table 的独立重算——
保证与产生轨迹的环境价格逐项一致（wrapper 播种顺序修复后两者随机流
起点不同，独立重算不再可靠）。

用法:
  WEBSHOP_ITEM_PATH=... WEBSHOP_ATTR_PATH=... WEBSHOP_HUMAN_INS_PATH=... \
  python scripts/mine_positions.py --traj-dirs data_out/clean_debug10_frozen \
      data_out/clean_baseline_100a data_out/clean_baseline_100b
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.env_wrapper import EnvConfig, WebShopEnvWrapper  # noqa: E402
from attack.positions import mine_all  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--max-per-task", type=int, default=1)
    ap.add_argument("--entry", default="item_page",
                    choices=["item_page", "description"],
                    help="注入入口；item_page=v1.1（agent 自然行为可达）")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data_out/attack/positions")
    args = ap.parse_args()

    traj_paths = []
    for d in args.traj_dirs:
        traj_paths += sorted(p for p in d.glob("goal_*.json"))
    if not traj_paths:
        print("未找到轨迹文件", file=sys.stderr)
        return 1
    print(f"[mine] {len(traj_paths)} 条 clean 轨迹", flush=True)

    cfg = EnvConfig(seed=233, human_goals=True, num_products=None)
    wrapper = WebShopEnvWrapper(cfg)
    products = wrapper.product_table()
    print(f"[mine] env 商品表 {len(products)} 条", flush=True)

    summary = mine_all(traj_paths, products, args.max_per_task, entry=args.entry)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "positions.jsonl", "w", encoding="utf-8") as f:
        for p in summary["positions"]:
            f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
    with open(args.out_dir / "excluded.jsonl", "w", encoding="utf-8") as f:
        for r in summary["results"]:
            if r.exclusion:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    print(f"任务 {summary['n_tasks']} -> 合格位置 {summary['n_positions']}")
    print(f"排除分布: {json.dumps(summary['exclusions'], ensure_ascii=False)}")
    print(f"输出: {args.out_dir}/positions.jsonl, excluded.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
