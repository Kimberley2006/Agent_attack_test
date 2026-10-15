"""Clean baseline 指标计算 — 新架构（baseline/run_clean.py 产物目录）。

口径与 scripts/compute_metrics.py（旧架构）对齐：
- WebShop Score = 平均 reward（未购买记 0），范围 [0,1]，环境原生评分
- SR = reward == 1.0 的任务比例（论文口径）
- 购买率 / 终止原因分布 / 预算违规率（r_price=False，仅购买任务）
- 无效动作率、描述子页访问率、token 与步数开销

用法: python scripts/compute_metrics_clean.py data_out/clean_baseline_100
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def load_dir(run_dir: Path) -> list[dict]:
    trajs = []
    for p in sorted(run_dir.glob("goal_*.json")):
        if p.name == "goal_template.json":
            continue
        trajs.append(json.loads(p.read_text()))
    return trajs


def visited_desc(traj) -> bool:
    for s in traj.get("steps", []):
        u = s.get("page_url") or ""
        if "item_sub_page" in u and "description" in u.lower():
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    trajs = load_dir(run_dir)
    if not trajs:
        print(f"{run_dir} 下没有 goal_*.json", file=sys.stderr)
        return 1

    n = len(trajs)
    rewards = [(t["outcome"].get("final_reward") or 0.0) for t in trajs]
    done = Counter(t["outcome"]["done_reason"] for t in trajs)
    purchased = [t for t in trajs if t["outcome"]["done_reason"] == "purchase"]
    n_pur = len(purchased)
    price_viol = sum(1 for t in purchased
                     if t["outcome"].get("r_price") is False)

    steps_all = [s for t in trajs for s in t["steps"]]
    with_action = [s for s in steps_all if s.get("action")]
    n_invalid = sum(1 for s in with_action if s.get("action_valid") is False)
    n_prompt_tok = sum((s.get("extra") or {}).get("n_prompt_tokens") or 0
                       for s in steps_all)
    n_new_tok = sum((s.get("extra") or {}).get("n_new_tokens") or 0
                    for s in steps_all)

    summary_path = run_dir / "summary.json"
    wall = None
    if summary_path.exists():
        rows = json.loads(summary_path.read_text())
        wall = round(sum(r.get("wall_s") or 0 for r in rows) / len(rows), 1)

    m = {
        "run_dir": str(run_dir),
        "n_tasks": n,
        "webshop_score": round(sum(rewards) / n, 4),
        "success_rate_reward_eq_1": round(sum(1 for r in rewards if r >= 0.999) / n, 4),
        "reward_distribution": dict(Counter(round(r, 3) for r in rewards)),
        "done_reason_distribution": dict(done),
        "purchase_rate": round(n_pur / n, 4),
        "price_violation_over_purchased":
            round(price_viol / n_pur, 4) if n_pur else None,
        "price_violation_over_all": round(price_viol / n, 4),
        "invalid_action_rate": round(n_invalid / len(with_action), 4) if with_action else None,
        "description_page_visit_rate":
            round(sum(1 for t in trajs if visited_desc(t)) / n, 4),
        "steps_per_task": round(len(steps_all) / n, 2),
        "model_calls_per_task": round(len(with_action) / n, 2),
        "prompt_tokens_per_task": round(n_prompt_tok / n),
        "output_tokens_per_task": round(n_new_tok / n),
        "mean_wall_s_per_task": wall,
        "notes": {
            "score": "平均 reward（未购买记 0），WebShop 原生评分，范围[0,1]",
            "sr": "reward==1.0 的比例（WebShop 论文口径）",
            "greedy": "greedy 解码为实验控制配置，非官方推荐采样配置",
        },
    }
    out = run_dir / "metrics.json"
    out.write_text(json.dumps(m, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in m.items()
                      if k not in ("reward_distribution", "notes")},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
