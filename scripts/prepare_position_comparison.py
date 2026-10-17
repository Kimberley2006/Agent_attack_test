"""冻结 20 任务，选择后续不同商品页和自然到达的子页，并验证重放。

只读取 clean 轨迹，不调用模型。输出按位置分组的 JSONL 与完整性清单。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.env_wrapper import EnvConfig, WebShopEnvWrapper
from agent.schema import TaskTrajectory
from attack.positions import mine_trajectory
from attack.positions import _asin_from_url
from attack.inject import classify_page, parse_segments


def observed_products(traj):
    """离线候选表：使用页面显示价格；真实环境价格仍需重放确认。"""
    products = {}
    for step in traj.steps:
        if "/item_page/" not in (step.page_url or "") or classify_page(step.obs) != "item_page":
            continue
        asin = _asin_from_url(step.page_url)
        segments = parse_segments(step.obs)
        price_index = next(i for i, text in enumerate(segments) if text.startswith("Price:"))
        values = re.findall(r"\$([\d,.]+)", segments[price_index])
        price = float(values[0].replace(",", ""))
        products[asin] = {"title": segments[price_index - 1], "price": price,
                          "displayed_price": segments[price_index], "available": True}
    return products


def select_positions(traj, products, first):
    """同一商品主页面只取首次访问；子页每种类型/商品只取首次进入。"""
    mined = mine_trajectory(traj, products, len(traj.steps), entry="item_page")
    if not mined.positions or mined.positions[0].position_id != first["position_id"]:
        raise ValueError(f"首位置漂移: {traj.task_id}")
    if mined.positions[0].obs_raw != first["obs_raw"]:
        raise ValueError(f"首位置观测漂移: {traj.task_id}")
    rows = []
    seen = set()
    ranks = {}
    for pos in mined.positions:
        if pos.asin in seen:
            continue
        seen.add(pos.asin)
        ranks[pos.asin] = len(seen)
        row = dict(first) if len(seen) == 1 else pos.to_dict()
        row.update(entry="item_page", item_visit_rank=len(seen),
                   position_group="first_item" if len(seen) == 1 else "later_item")
        rows.append(row)
    for entry in ("features", "description", "reviews"):
        seen_subpage = set()
        for pos in mine_trajectory(traj, products, len(traj.steps), entry=entry).positions:
            # 配对页面类型比较要求同商品主页面也在所选位置中。
            if pos.asin not in ranks or pos.asin in seen_subpage:
                continue
            seen_subpage.add(pos.asin)
            row = pos.to_dict()
            row.update(entry=entry, item_visit_rank=ranks[pos.asin],
                       position_group=entry)
            rows.append(row)
    return sorted(rows, key=lambda r: r["step_index"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path,
                    default=ROOT / "configs/position_comparison20.json")
    ap.add_argument("--offline", action="store_true",
                    help="仅依据保存的轨迹选点；标记环境重放和真实价格尚未验证")
    args = ap.parse_args()
    config = json.loads(args.config.read_text())
    source = ROOT / config["source_positions"]
    source_bytes = source.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != config["source_sha256"]:
        raise ValueError("原始位置清单指纹变化")
    originals = [json.loads(line) for line in source_bytes.decode().splitlines() if line.strip()]
    selected = originals[:config["n_tasks"]]
    if [r["task_id"] for r in selected] != config["task_ids"]:
        raise ValueError("任务顺序与冻结配置不符")
    wrapper = None if args.offline else WebShopEnvWrapper(EnvConfig(seed=233, human_goals=True, num_products=None))
    products = wrapper.product_table() if wrapper is not None else None
    all_rows = []
    trajectory_hashes = {}
    for first in selected:
        paths = [ROOT / d / f"{first['task_id']}.json" for d in config["clean_dirs"]]
        path = next((p for p in paths if p.exists()), None)
        if path is None:
            raise FileNotFoundError(first["task_id"])
        trajectory_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        traj = TaskTrajectory.load(path)
        if wrapper is not None:
            goal = wrapper.goal(int(first["task_id"].removeprefix("goal_")))
            if goal != traj.goal:
                raise ValueError(f"任务定义漂移: {traj.task_id}")
        rows = select_positions(traj, observed_products(traj) if args.offline else products, first)
        for row in rows:
            k = row["step_index"]
            if row["replay_actions"] != [s.action for s in traj.steps[:k]]:
                raise ValueError(f"动作前缀不匹配: {row['position_id']}")
            messages = traj.steps[k].messages
            if not messages or messages[-1]["role"] != "user" or not messages[-1]["content"].endswith(row["obs_raw"]):
                raise ValueError(f"分支消息缺失或观测不匹配: {row['position_id']}")
        # 每个任务只需重放至最晚位置；同时验证所有较早位置的观测。
        if wrapper is not None:
            replay = wrapper.replay(int(first["task_id"].removeprefix("goal_")), rows[-1]["replay_actions"])
            for row in rows:
                if replay[row["step_index"]].obs != row["obs_raw"]:
                    raise ValueError(f"重放观测不一致: {row['position_id']}")
        for row in rows:
            row["replay_verified"] = wrapper is not None
            row["price_source"] = "original_position" if row["position_group"] == "first_item" else "unverified" if args.offline else "environment"
            if args.offline and row["position_group"] != "first_item":
                row["displayed_price"] = observed_products(traj)[row["asin"]]["displayed_price"]
                row["item_price"] = None
                row["violation_type"] = "wrong_item"  # 精确预算违规待环境价格确认
        all_rows.extend(rows)
        print(f"{traj.task_id}: {len(rows)} 个位置，{'离线检查通过，待重放' if args.offline else '重放通过'}", flush=True)
    groups = {
        "positions": [r for r in all_rows if r["position_group"] == "first_item"],
        "additional_positions": [r for r in all_rows if r["position_group"] != "first_item"],
        "all_positions": all_rows,
    }
    out = ROOT / config["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    file_hashes = {}
    for name, rows in groups.items():
        path = out / f"{name}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        file_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    additional = groups["additional_positions"]
    report = {
        "config": config,
        "validation": "offline messages checked; environment replay and exact prices pending" if args.offline else "all selected goals, messages and replay observations matched",
        "n_tasks": len(selected), "n_first_positions": len(groups["positions"]),
        "n_additional_positions": len(additional), "n_total_positions": len(all_rows),
        "groups": dict(Counter(r["position_group"] for r in all_rows)),
        "tasks_with_additional_positions": list(dict.fromkeys(r["task_id"] for r in additional)),
        "files_sha256": file_hashes, "trajectory_sha256": trajectory_hashes,
        "positions": [{k: r[k] for k in ("position_id", "task_id", "step_index", "asin", "entry", "position_group", "item_visit_rank")} for r in all_rows],
    }
    (out / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("n_tasks", "n_first_positions", "n_additional_positions", "n_total_positions", "groups")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
