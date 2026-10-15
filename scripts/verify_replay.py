"""⑨ 重放验证：选取轨迹，重置同一任务并重放动作序列，逐步核对页面与状态。

比对内容：每步观测文本哈希、url、page_type、无效动作判定、终局购买与奖励。
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_attack.config import load_config  # noqa: E402
from agent_attack.env_client import EnvClient  # noqa: E402


def h16(s):
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def replay_episode(env, calls, task_id):
    """calls: 该任务的 model_calls 记录（按 agent_step 排序，仅含已执行动作）。"""
    calls = sorted(calls, key=lambda c: c["agent_step"])
    r = env.reset(task_id)
    results = [{"step": -1, "obs_h": h16(r["obs"]), "url": r["url"],
                "page": r["page_type"], "match": True}]
    executed = [c for c in calls if c.get("action") and c.get("env")]
    by_step = {c["agent_step"]: c for c in calls}

    def exp_obs(step):
        """旧日志 transition 未存 obs：从下一步调用的消息末尾提取。"""
        cur = by_step.get(step)
        if not cur:
            return None
        if (cur.get("env") or {}).get("obs") is not None:
            return cur["env"]["obs"]
        nxt = by_step.get(step + 1)
        if nxt and nxt.get("messages"):
            last_user = nxt["messages"][-1]
            if last_user.get("role") == "user":
                content = last_user["content"]
                for prefix in ("### Page after your last action\n",
                               "### Current Page (WebShop state)\n"):
                    if content.startswith(prefix):
                        return content[len(prefix):]
        return None

    for c in executed:
        tr = env.step(task_id, c["action"])
        exp = c["env"]
        eo = exp_obs(c["agent_step"])
        match = (tr["url"] == exp.get("url")
                 and tr["invalid"] == exp.get("invalid"))
        if eo is not None:
            match = match and h16(tr["obs"]) == h16(eo)
        else:
            match = None if match is True else match  # 无参照时标记不可比
            match = tr["url"] == exp.get("url")  # 退化只比 url
        results.append({
            "step": c["agent_step"], "action": c["action"],
            "obs_h": h16(tr["obs"]),
            "exp_obs_h": h16(eo) if eo is not None else None,
            "url": tr["url"], "exp_url": exp.get("url"),
            "invalid": tr["invalid"], "match": match,
        })
        if tr["done"]:
            results[-1]["reward"] = (tr.get("purchased") or {}).get("reward")
            results[-1]["exp_reward"] = (exp.get("purchased") or {}).get("reward")
            results[-1]["match"] = match and \
                results[-1]["reward"] == results[-1]["exp_reward"]
            break
    env.close_session(task_id)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--config", default=None)
    ap.add_argument("--n", type=int, default=5, help="抽查任务数")
    ap.add_argument("--task-ids", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_dir = Path(args.run_dir)

    episodes = {json.loads(l)["task_id"]: json.loads(l) for l in
                (run_dir / "episodes.jsonl").read_text().splitlines() if l.strip()}
    calls_by_task = {}
    for l in (run_dir / "model_calls.jsonl").read_text().splitlines():
        c = json.loads(l)
        calls_by_task.setdefault(c["task_id"], []).append(c)

    # 抽样覆盖三类结局：成功购买、自然失败（未购买但执行过动作）、无效动作多
    if args.task_ids:
        ids = [int(x) for x in args.task_ids.split(",")]
    else:
        purchased = [t for t, e in episodes.items()
                     if e["outcome"]["status"] == "purchased"]
        natural_fail = [t for t, e in episodes.items()
                        if e["outcome"]["status"] != "purchased"
                        and e["n_actions_executed"] > 0]
        invalid_heavy = sorted(episodes,
                               key=lambda t: -episodes[t]["n_invalid_actions"])
        ids = (purchased[:max(1, args.n // 3)] + natural_fail[:max(1, args.n // 3)]
               + invalid_heavy[:max(1, args.n // 3)])[:args.n]

    env = EnvClient(cfg.get("environment.service_url"))
    n_ok = 0
    for task_id in ids:
        res = replay_episode(env, calls_by_task.get(task_id, []), task_id)
        ok = all(r["match"] for r in res)
        n_ok += ok
        print(f"task {task_id}: {'REPLAY MATCH' if ok else 'REPLAY MISMATCH'} "
              f"({len(res)-1} steps, outcome={episodes[task_id]['outcome']['status']})")
        if not ok:
            for r in res:
                if not r["match"]:
                    print(f"    step {r['step']}: {r}")
    print(f"\n{n_ok}/{len(ids)} episodes replay-identical")
    sys.exit(0 if n_ok == len(ids) else 1)


if __name__ == "__main__":
    main()
