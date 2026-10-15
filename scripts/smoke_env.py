# -*- coding: utf-8 -*-
"""WebShop 进程内环境冒烟测试（阶段 0）。

验证四件事：
1. WebAgentTextEnv 能在 webshop conda 环境中创建并交互（BM25 回退检索）；
2. 描述子页观测确实是 5 段结构（注入器的前提）；
3. goal 的 price_upper 与商品价格语义正确（违规判定的前提）；
4. 固定 random seed 后，两个独立 env 实例重放同一动作序列，
   逐步观测与关键状态逐字节一致（配对分支重放的前提）。

用法: /home/ma-user/conda-envs/webshop/bin/python scripts/smoke_env.py
"""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                       # Agent_Attack（attack 包）
sys.path.insert(0, str(ROOT / "webshop_repo"))      # web_agent_site

from web_agent_site.envs import WebAgentTextEnv  # noqa: E402

from attack.inject import classify_page, parse_segments  # noqa: E402

SEED = 42
NON_PRODUCT_WORDS = {
    "back to search", "< prev", "next >", "> next", "buy now",
    "description", "features", "reviews", "search", "attributes",
}
_ASIN_RE = re.compile(r"/item_page/[^/]+/([A-Z0-9]{10})")
_PRODUCT_LINK_RE = re.compile(r"^[a-z0-9]{10}$")   # 结果页商品链接文本是小写 asin


def make_env() -> WebAgentTextEnv:
    # 价格 uniform 抽样、goal price_upper 抽样、goal 洗牌全部发生在 env 创建期间，
    # 必须在创建前播种（SimServer 内部的 seed(233) 只覆盖之后的洗牌）。
    random.seed(SEED)
    return WebAgentTextEnv(
        observation_mode="text",
        num_products=1000,
        human_goals=1,
        get_image=False,
        session_prefix="smoke",
    )


def pick_goal(env: WebAgentTextEnv) -> int:
    for i, g in enumerate(env.server.goals):
        if 0 < g["price_upper"] < 100000:
            return i
    raise RuntimeError("没有带价格约束的 goal")


def run_episode(env: WebAgentTextEnv, goal_idx: int) -> dict:
    trace = {"obs": [], "urls": [], "actions": [], "info": {}}
    obs = env.reset(session=goal_idx)
    trace["obs"].append(obs)
    trace["urls"].append(env.browser.current_url)
    goal = env.server.goals[goal_idx]

    # 1) 按任务 query 搜索（先执行，商品链接在结果页才出现）
    obs, reward, done, info = env.step(f"search[{goal['query']}]")
    trace["actions"].append(f"search[{goal['query']}]")
    trace["obs"].append(obs)
    trace["urls"].append(env.browser.current_url)

    # 2) 点第一个真实商品链接（结果页的 product-link 文本是 10 位小写 asin）
    clickables = env.get_available_actions()["clickables"]
    products = [c for c in clickables if _PRODUCT_LINK_RE.match(c)]
    assert products, f"搜索结果无商品链接: {clickables}"
    steps = [
        f"click[{products[0]}]",
        "click[description]",
        "click[< prev]",
        "click[buy now]",
    ]
    for a in steps:
        trace["actions"].append(a)
        obs, reward, done, info = env.step(a)
        trace["obs"].append(obs)
        trace["urls"].append(env.browser.current_url)
        if done:
            trace["info"] = {"reward": reward, "done": done}
            break

    item_url = trace["urls"][2]
    m = _ASIN_RE.search(item_url)
    asin = m.group(1) if m else None
    trace["info"].update(
        asin=asin,
        item_price=env.server.product_prices.get(asin) if asin else None,
        price_upper=goal["price_upper"],
        instruction_text=goal["instruction_text"],
        desc_obs=trace["obs"][3],
        desc_url=trace["urls"][3],
    )
    return trace


def main() -> int:
    print("[1/4] 创建第一个 env 实例（seed=%d）..." % SEED)
    env_a = make_env()
    goal_idx = pick_goal(env_a)
    trace_a = run_episode(env_a, goal_idx)

    ok = True

    # --- 结构断言 ---
    desc_obs = trace_a["info"]["desc_obs"]
    segs = parse_segments(desc_obs)
    struct_ok = (
        classify_page(desc_obs) == "description_like"
        and len(segs) == 5
        and segs[0] == "Instruction:"
        and segs[2] == "Back to Search"
        and segs[3] == "< Prev"
        and "/Description/" in trace_a["info"]["desc_url"]
    )
    print(f"[2/4] 描述子页结构: {'OK' if struct_ok else 'FAIL'} "
          f"(段数={len(segs)}, 描述长度={len(segs[-1])})")
    ok &= struct_ok

    # --- 语义断言 ---
    info = trace_a["info"]
    sem_ok = (
        info["asin"] is not None
        and info["item_price"] is not None
        and 0 < info["price_upper"] < 100000
        and "price lower than" in info["instruction_text"]
        and "reward" in info  # 购买触发终局
    )
    print(f"[3/4] 任务/价格语义: {'OK' if sem_ok else 'FAIL'} "
          f"(asin={info['asin']}, 价格={info['item_price']}, "
          f"预算={info['price_upper']}, reward={info.get('reward')})")
    ok &= sem_ok
    print(f"      指令: {info['instruction_text'][:90]}")

    # --- 重放确定性 ---
    print("[4/4] 第二个 env 实例重放...")
    env_b = make_env()
    goal_b = env_b.server.goals[goal_idx]
    trace_b = run_episode(env_b, goal_idx)
    same_goal = goal_b["instruction_text"] == info["instruction_text"] and \
        goal_b["price_upper"] == info["price_upper"]
    same_prices = env_a.server.product_prices == env_b.server.product_prices
    same_obs = trace_a["obs"] == trace_b["obs"]
    same_urls = trace_a["urls"] == trace_b["urls"]
    replay_ok = same_goal and same_prices and same_obs and same_urls
    print(f"      goal 一致: {same_goal}, 全量价格表一致: {same_prices}, "
          f"观测逐字节一致: {same_obs}, URL 一致: {same_urls}")
    if not same_obs:
        for i, (x, y) in enumerate(zip(trace_a["obs"], trace_b["obs"])):
            if x != y:
                print(f"      首个差异在 step {i}:")
                print(f"        A: {x[:160]!r}")
                print(f"        B: {y[:160]!r}")
                break
    ok &= replay_ok

    print("\n冒烟结果:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
