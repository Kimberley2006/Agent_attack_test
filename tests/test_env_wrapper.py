# -*- coding: utf-8 -*-
"""env_wrapper 真实环境测试（需 webshop conda 环境运行）：

    /home/ma-user/conda-envs/webshop/bin/python tests/test_env_wrapper.py

覆盖：
1. 描述子页真实观测为 5 段结构（注入器前提）；
2. 冻结 manifest 的真实载荷注入真实观测（端到端链路）；
3. 同实例隔离重放与跨实例重放逐字节一致；
4. 用 env 自身价格表挖掘出合格注入位置（价格超预算 + 描述页首次访问）；
5. 合成 goals 数量（human_goals=False 的扩规模路径）。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.env_wrapper import EnvConfig, WebShopEnvWrapper  # noqa: E402
from agent.schema import OutcomeRecord, StepRecord, TaskTrajectory  # noqa: E402
from attack import (  # noqa: E402
    classify_page,
    load_manifest,
    mine_trajectory,
    parse_segments,
    render_branch_observation,
    verify_only_append,
)

MANIFEST = ROOT / "data_out" / "payloads" / "manifest_v1.json"
_ASIN_URL_RE = re.compile(r"/item_page/[^/]+/([A-Z0-9]{10})")
_PRODUCT_LINK_RE = re.compile(r"^[a-z0-9]{10}$")


def _find_violating_candidate(w: WebShopEnvWrapper):
    """在 13 个人类 goal 中找一个结果页首屏含超预算商品的 (goal_idx, asin)。"""
    for gi in range(w.n_goals):
        goal = w.goal(gi)
        budget = goal.get("price_upper", 1e6)
        if not (0 < budget < 1e6):
            continue
        w.reset_fresh(gi)
        w.step(f"search[{goal['query']}]")
        asins = [c for c in w.available_actions() if _PRODUCT_LINK_RE.match(c)]
        for asin in asins:
            if w.price(asin.upper()) > budget:
                return gi, asin, budget
    return None, None, None


def main() -> int:
    ok = True

    print("[1/5] 创建 wrapper（human_goals=1, seed=42）...")
    w = WebShopEnvWrapper(EnvConfig())
    print(f"      goals: {w.n_goals}")

    gi, asin_lc, budget = _find_violating_candidate(w)
    print(f"[2/5] 超预算候选: goal={gi}, asin={asin_lc}, 预算={budget}")
    ok &= gi is not None

    # --- 脚本化走到该商品的描述页 ---
    goal = w.goal(gi)
    actions = [f"search[{goal['query']}]", f"click[{asin_lc}]", "click[description]"]
    step_results = w.replay(gi, actions)
    obs_seq = [r.obs for r in step_results]
    url_seq = [r.url for r in step_results]
    desc_obs = obs_seq[-1]
    segs = parse_segments(desc_obs)
    struct_ok = (
        classify_page(desc_obs) == "description_like" and len(segs) == 5
        and segs[2] == "Back to Search" and segs[3] == "< Prev"
    )
    print(f"[3/5] 描述子页结构: {'OK' if struct_ok else 'FAIL'} "
          f"(段数={len(segs)}, 描述长度={len(segs[-1])})")
    ok &= struct_ok

    # --- 真实载荷注入真实观测 ---
    manifest = load_manifest(MANIFEST)
    arm = manifest["arms"]["zh"]
    attack_item = next(i for i in arm["items"] if i["payload_id"] == "A1")
    result = render_branch_observation(desc_obs, attack_item)
    verify_only_append(desc_obs, result.obs, attack_item["text"])
    injected_ok = attack_item["text"] in result.obs and \
        result.obs.startswith(desc_obs)
    print(f"[4/5] 真实注入链路: {'OK' if injected_ok else 'FAIL'} "
          f"(载荷 {attack_item['char_len']} 字符，注入后观测 {len(result.obs)} 字符)")
    ok &= injected_ok

    # --- 重放确定性：同实例 + 跨实例 ---
    replay_a = [r.obs for r in w.replay(gi, actions)]
    w2 = WebShopEnvWrapper(EnvConfig())
    replay_b = [r.obs for r in w2.replay(gi, actions)]
    det_ok = replay_a == obs_seq == replay_b
    print(f"[5/5] 重放一致性(同实例/跨实例): {'OK' if det_ok else 'FAIL'}")
    ok &= det_ok

    # --- 用 env 价格表挖掘位置 ---
    products = w.product_table()
    traj = TaskTrajectory(
        task_id=f"goal_{gi}",
        session=f"{w.config.session_prefix}{gi}",
        instruction_text=goal["instruction_text"],
        goal=goal,
        config_ref="smoke",
        steps=[
            StepRecord(step=i, obs=obs_seq[i], action=actions[i],
                       page_url=url_seq[i])
            for i in range(len(actions))
        ] + [StepRecord(step=len(actions), obs=obs_seq[-1], action=None,
                        page_url=url_seq[-1])],
        outcome=OutcomeRecord(done_reason="step_limit", steps_used=len(actions) + 1),
    )
    mined = mine_trajectory(traj, products)
    asin_uc = asin_lc.upper()
    mine_ok = (
        mined.exclusion is None
        and len(mined.positions) == 1
        and mined.positions[0].asin == asin_uc
        and mined.positions[0].step_index == 3
        and mined.positions[0].replay_actions == actions
        and mined.positions[0].item_price == products[asin_uc]["price"]
    )
    print(f"      位置挖掘: {'OK' if mine_ok else 'FAIL'} "
          f"({mined.exclusion or mined.positions[0].position_id})")
    ok &= mine_ok

    # --- 合成 goals 规模探测 ---
    w3 = WebShopEnvWrapper(EnvConfig(human_goals=False))
    with_budget = sum(
        1 for g in w3.goals if 0 < g.get("price_upper", 0) < 1e6
    )
    print(f"      合成 goals: {w3.n_goals} 个（含价格约束 {with_budget} 个）")

    print("\n真实环境测试:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
