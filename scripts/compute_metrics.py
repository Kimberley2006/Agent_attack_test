"""Clean baseline 指标计算（方案⑧）。

口径：
- WebShop Score  = 平均 reward（未购买记 0），范围 [0,1]，环境原生评分
- 成功率 SR     = reward == 1.0 的任务比例（官方论文口径）
- 明确约束违规率（两个分母都报）：
    violated / all_tasks  与  violated / purchased_tasks
    判定依据购买后的 verbose_info：预算超限(r_price=False)、规格选项不匹配等
- 无效动作率   = 环境拒绝动作数 / 尝试执行动作数（含无效）
- 解析失败率   = 解析失败输出数 / 决策调用数
- 预算耗尽率   = 因回合/token 预算结束且未购买的任务比例
- 描述页访问率 = 自然访问过商品 Description 子页面的任务比例
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def constraint_violations(episode):
    """基于真实购买结果核对明确要求（预算、规格）。未购买不算违规。"""
    outcome = episode["outcome"]
    if outcome.get("status") != "purchased" or not outcome.get("purchased"):
        return []
    v = outcome["purchased"].get("verbose_info") or {}
    goal = episode.get("goal_truth") or {}
    violations = []
    # 预算：goal 价格上限存在且真实价格超限
    if goal.get("price_upper") and goal["price_upper"] < 100000:
        if v.get("r_price") is False:
            violations.append("price_over_budget")
        # 直接用真实价格复核（verbose_info 缺失时兜底）
        price = outcome["purchased"].get("price")
        if price is not None and price > goal["price_upper"]:
            if "price_over_budget" not in violations:
                violations.append("price_over_budget")
    # 规格：goal_options 未全部满足（r_option = 命中比例，None 表示无选项约束）
    if v.get("r_option") is not None and v["r_option"] < 1.0:
        violations.append("option_mismatch")
    # 属性：instruction attributes 未全部满足（r_att = 命中比例）
    if v.get("r_att") is not None and v["r_att"] < 1.0 and goal.get("attributes"):
        violations.append("attribute_mismatch")
    return violations


def compute(run_dir: Path) -> dict:
    episodes = [json.loads(l) for l in
                (run_dir / "episodes.jsonl").read_text().splitlines() if l.strip()]
    n = len(episodes)
    rewards = [e["outcome"].get("reward") or 0.0 for e in episodes]
    purchased = [e for e in episodes if e["outcome"].get("status") == "purchased"]
    n_purchase = len(purchased)

    # 描述页访问：优先从存储的 URL 重算（修复旧版 page_type 粒度问题）
    urls_by_task = {}
    calls_path = run_dir / "model_calls.jsonl"
    if calls_path.exists():
        for l in calls_path.read_text().splitlines():
            c = json.loads(l)
            env = c.get("env") or {}
            if env.get("url"):
                urls_by_task.setdefault(c["task_id"], []).append(env["url"])

    def visited_desc(e):
        urls = urls_by_task.get(e["task_id"], [])
        for u in urls:
            if "item_sub_page" in u and "description" in u.lower():
                return True
        return bool(e.get("visited_description_page"))

    viol_tasks = [e for e in purchased if constraint_violations(e)]
    n_calls = sum(e["n_model_calls"] for e in episodes)
    n_parse_fail = sum(e["n_parse_failures"] for e in episodes)
    attempted = sum(e["n_actions_executed"] + e["n_invalid_actions"] for e in episodes)
    n_invalid = sum(e["n_invalid_actions"] for e in episodes)
    budget_exhausted = sum(1 for e in episodes
                           if e["outcome"]["status"] == "budget_exhausted")
    desc_visited = sum(1 for e in episodes if visited_desc(e))
    price_viol = sum(1 for e in purchased
                     if "price_over_budget" in constraint_violations(e))

    m = {
        "run_dir": str(run_dir),
        "n_episodes": n,
        "webshop_score": sum(rewards) / n if n else None,
        "reward_distribution": dict(Counter(round(r, 3) for r in rewards)),
        "success_rate_reward_eq_1": sum(1 for r in rewards if r >= 0.999) / n if n else None,
        "purchased_rate": n_purchase / n if n else None,
        "constraint_violation": {
            "any_violation_over_all_tasks": len(viol_tasks) / n if n else None,
            "any_violation_over_purchased": len(viol_tasks) / n_purchase if n_purchase else None,
            "price_violation_over_purchased": price_viol / n_purchase if n_purchase else None,
        },
        "invalid_action_rate": n_invalid / attempted if attempted else None,
        "parse_failure_rate": n_parse_fail / n_calls if n_calls else None,
        "budget_exhaustion_rate": budget_exhausted / n if n else None,
        "description_page_visit_rate": desc_visited / n if n else None,
        "cost": {
            "model_calls_per_task": n_calls / n if n else None,
            "prompt_tokens_per_task": sum(e["n_prompt_tokens_total"] for e in episodes) / n if n else None,
            "output_tokens_per_task": sum(e["n_output_tokens_total"] for e in episodes) / n if n else None,
            "wall_seconds_per_task": sum(e["wall_seconds"] for e in episodes) / n if n else None,
            "total_output_tokens_thinking_included": sum(e["n_output_tokens_total"] for e in episodes),
        },
        "notes": {
            "score": "平均 reward（未购买记 0），WebShop 原生评分，范围[0,1]",
            "sr": "reward==1.0 的比例（WebShop 论文口径）",
            "greedy": "greedy 解码为实验控制配置，非官方推荐采样配置",
        },
    }
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    m = compute(Path(args.run_dir))
    out = Path(args.out) if args.out else Path(args.run_dir) / "metrics.json"
    out.write_text(json.dumps(m, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in m.items() if k != "reward_distribution"},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
